"""Real-image drone detector: fine-tune YOLO on real photographs.

The synthetic/3D detectors (detect.py, render3d.py) answer "does the pipeline
work"; this answers "does it hold on real imagery." Fine-tunes a pretrained
yolo11n on the pathikg/drone-detection-dataset (2625 real drone photos, COCO
bboxes, single 'drone' class) pulled from Hugging Face, then measures the same
thing the synthetic curve does — detection rate vs apparent size — but on real
data, where small/distant drones are the hard case.

    python yolo_real.py prepare     # parquet -> yolo_ds/ (images + YOLO labels)
    python yolo_real.py train [ep]  # fine-tune yolo11n on GPU -> runs/drone/
    python yolo_real.py curve       # recall vs bbox-height + sample predictions

Needs: realdata/test.parquet (HF, ~275 MB), ultralytics, CUDA torch.
ponytail: uses only the 288 MB test shard (2625 imgs) — plenty to fine-tune a
pretrained net; pull more shards if mAP is data-starved.
"""

import sys
from pathlib import Path

import numpy as np

DS = Path("yolo_ds")
PARQUET = Path("realdata/test.parquet")


def prepare():
    import pyarrow.parquet as pq
    rows = pq.read_table(PARQUET).to_pylist()
    for s in ("train", "val"):
        (DS / "images" / s).mkdir(parents=True, exist_ok=True)
        (DS / "labels" / s).mkdir(parents=True, exist_ok=True)
    n_obj = 0
    for i, r in enumerate(rows):
        split = "val" if i % 7 == 0 else "train"           # ~14% val
        (DS / "images" / split / f"{i}.jpg").write_bytes(r["image"]["bytes"])
        w_img, h_img = r["width"], r["height"]
        lines = []
        for box in (r.get("objects") or {}).get("bbox") or []:
            x, y, bw, bh = box                              # COCO: top-left x,y,w,h
            xc = min(max((x + bw / 2) / w_img, 0.0), 1.0)
            yc = min(max((y + bh / 2) / h_img, 0.0), 1.0)
            nw, nh = min(bw / w_img, 1.0), min(bh / h_img, 1.0)
            if nw > 0 and nh > 0:
                lines.append(f"0 {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")
                n_obj += 1
        (DS / "labels" / split / f"{i}.txt").write_text("\n".join(lines))
    (DS / "data.yaml").write_text(
        f"path: {DS.resolve().as_posix()}\ntrain: images/train\nval: images/val\nnames:\n  0: drone\n")
    n_val = len(list((DS / "images" / "val").glob("*.jpg")))
    print(f"prepared {len(rows)} images ({n_val} val), {n_obj} drone boxes -> {DS}/")


def train(epochs=40):
    from ultralytics import YOLO
    m = YOLO("yolo11n.pt")   # pretrained COCO nano, auto-downloads (~5 MB)
    m.train(data=str(DS / "data.yaml"), epochs=epochs, imgsz=640, batch=16,
            device=0, project="runs", name="drone", exist_ok=True, verbose=True)
    print("trained -> runs/drone/weights/best.pt")


def _iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / (ua + 1e-9)


def curve(weights=None):
    from ultralytics import YOLO
    if weights is None:                      # ultralytics nests the run dir; find newest best.pt
        cands = sorted(Path("runs").rglob("best.pt"), key=lambda p: p.stat().st_mtime)
        weights = str(cands[-1]) if cands else "runs/detect/runs/drone/weights/best.pt"
    m = YOLO(weights)
    val = sorted((DS / "images" / "val").glob("*.jpg"))

    heights, hits = [], []
    for im in val:
        lab = DS / "labels" / "val" / f"{im.stem}.txt"
        if not lab.exists() or not lab.read_text().strip():
            continue
        res = m.predict(str(im), conf=0.25, verbose=False)[0]
        H, W = res.orig_shape
        preds = res.boxes.xyxy.cpu().numpy() if len(res.boxes) else np.zeros((0, 4))
        for line in lab.read_text().splitlines():
            _, xc, yc, w, h = map(float, line.split())
            gx0, gy0 = (xc - w / 2) * W, (yc - h / 2) * H
            gx1, gy1 = (xc + w / 2) * W, (yc + h / 2) * H
            det = any(_iou((gx0, gy0, gx1, gy1), tuple(p)) > 0.3 for p in preds)
            heights.append(h * H)
            hits.append(det)
    heights, hits = np.array(heights), np.array(hits)

    edges = [0, 15, 25, 40, 60, 90, 140, 400]
    mids, rates, ns = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m_ = (heights >= lo) & (heights < hi)
        if m_.sum() >= 5:
            mids.append((lo + hi) / 2)
            rates.append(hits[m_].mean())
            ns.append(int(m_.sum()))
    print(f"{len(heights)} val drones; overall recall {hits.mean():.2f}")
    for md, rt, n in zip(mids, rates, ns):
        print(f"  ~{md:4.0f} px : {rt:.2f}  (n={n})")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(mids, rates, "o-", lw=2, color="tab:red")
    for md, rt, n in zip(mids, rates, ns):
        ax.annotate(f"n={n}", (md, rt), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8)
    ax.set(xlabel="drone bbox height (px)  — smaller = farther", ylabel="recall @IoU0.3",
           title="Real-image drone detector (YOLO11n): recall vs apparent size", ylim=(0, 1.02))
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("real_detection.png", dpi=120)

    # sample predictions montage
    fig2, axes = plt.subplots(2, 3, figsize=(12, 7))
    for ax, im in zip(axes.ravel(), val[:6]):
        res = m.predict(str(im), conf=0.25, verbose=False)[0]
        ax.imshow(res.plot()[:, :, ::-1])   # BGR->RGB
        ax.axis("off")
    fig2.suptitle("YOLO11n on real drone photos (val)")
    fig2.tight_layout()
    fig2.savefig("real_samples.png", dpi=110)
    print("saved real_detection.png + real_samples.png")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "prepare"
    if cmd == "train":
        train(int(sys.argv[2]) if len(sys.argv) > 2 else 40)
    elif cmd == "curve":
        curve()
    else:
        prepare()
