"""moderngl 3D drone renderer — the 'realistic image' path.

Renders a lit 3D quadcopter (body + crossed arms + 4 rotor disks) in the SAME
pinhole projection detect.py uses, so a real perspective-correct drone lands
exactly where project() says it should, and its apparent size falls off with
range for free (true 1/R perspective, not a hand-tuned blob). Returns a grayscale
intensity layer + coverage mask that detect.render_frame_3d composites over the
domain-randomized background.

Headless-safe: moderngl.create_standalone_context() makes its own GL context
(verified on Windows + NVIDIA). One context/program/FBO, reused across frames.

    python render3d.py test    # context + projection-match + size-falloff checks

ponytail: the quad is rendered level (no banking) — a real attitude model would
rotate the mesh by the velocity/bank, add it if the detector must key on pose.
"""

import itertools
import sys

import numpy as np

# canonical cube: 36 vertices (2 tris x 6 faces), positions in [-1,1] + face normals
_FACES = [
    ([(-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)], (0, 0, 1)),
    ([(1, -1, -1), (-1, -1, -1), (-1, 1, -1), (1, 1, -1)], (0, 0, -1)),
    ([(1, -1, 1), (1, -1, -1), (1, 1, -1), (1, 1, 1)], (1, 0, 0)),
    ([(-1, -1, -1), (-1, -1, 1), (-1, 1, 1), (-1, 1, -1)], (-1, 0, 0)),
    ([(-1, 1, 1), (1, 1, 1), (1, 1, -1), (-1, 1, -1)], (0, 1, 0)),
    ([(-1, -1, -1), (1, -1, -1), (1, -1, 1), (-1, -1, 1)], (0, -1, 0)),
]

VS = """
#version 330
in vec3 in_pos; in vec3 in_norm;
uniform vec3 dronepos, cright, cup, cfwd;
uniform float fnorm, nearp, scale;
out vec3 vnorm;
void main(){
    vec3 p = in_pos * scale + dronepos;
    float xc = dot(p, cright), yc = dot(p, cup), zc = dot(p, cfwd);
    gl_Position = vec4(2.0*fnorm*xc, 2.0*fnorm*yc, zc - 2.0*nearp, zc);
    vnorm = in_norm;
}
"""

FS = """
#version 330
in vec3 vnorm; out vec4 f;
uniform vec3 lightdir;
void main(){
    float d = max(dot(normalize(vnorm), normalize(lightdir)), 0.0);
    float shade = (0.12 + 0.55*d) * 0.55;   // dark body against bright sky
    f = vec4(vec3(shade), 1.0);
}
"""


def _box(center, half):
    c, h, tris, norms = np.array(center, float), np.array(half, float), [], []
    for quad, n in _FACES:
        q = [c + np.array(v, float) * h for v in quad]
        tris += [q[0], q[1], q[2], q[0], q[2], q[3]]
        norms += [n] * 6
    return tris, norms


def _disk(center, r, seg=16, z_up=0.0):
    c, tris, norms = np.array(center, float), [], []
    ring = [c + [r * np.cos(t), r * np.sin(t), 0] for t in np.linspace(0, 2 * np.pi, seg + 1)]
    for a, b in zip(ring[:-1], ring[1:]):
        tris += [c, a, b]
        norms += [(0, 0, 1)] * 3
    return tris, norms


def _quad_mesh():
    tris, norms = [], []
    for t, n in (_box([0, 0, 0], [0.30, 0.30, 0.10]),        # body
                 _box([0, 0, 0], [0.62, 0.07, 0.04]),        # arm bar
                 _box([0, 0, 0], [0.07, 0.62, 0.04])):       # arm bar
        tris += t; norms += n
    for dx, dy in [(0.52, 0), (-0.52, 0), (0, 0.52), (0, -0.52)]:  # rotor disks
        t, n = _disk([dx, dy, 0.08], 0.26)
        tris += t; norms += n
    return np.array(tris, "f4"), np.array(norms, "f4")


def _basis(az, el):
    ce = np.cos(el)
    fwd = np.array([ce * np.cos(az), ce * np.sin(az), np.sin(el)])
    right = np.cross(fwd, [0.0, 0.0, 1.0])
    n = np.linalg.norm(right)
    right = np.array([1.0, 0.0, 0.0]) if n < 1e-6 else right / n
    return fwd, right, np.cross(right, fwd)


class DroneRenderer:
    """One GL context/program/FBO, reused. render() -> (gray[img,img], mask[img,img])."""

    def __init__(self, img=64, fov_deg=90.0, size=1.6):
        self.img = img
        self.fnorm = 1.0 / (2.0 * np.tan(np.radians(fov_deg) / 2.0))
        self.size = size
        self.ctx = None

    def _init_gl(self):
        import moderngl
        self.ctx = moderngl.create_standalone_context()
        self.ctx.enable(moderngl.DEPTH_TEST)
        pos, norm = _quad_mesh()
        self.prog = self.ctx.program(vertex_shader=VS, fragment_shader=FS)
        vbo = self.ctx.buffer(np.hstack([pos, norm]).astype("f4").tobytes())
        self.vao = self.ctx.vertex_array(self.prog, [(vbo, "3f 3f", "in_pos", "in_norm")])
        self.fbo = self.ctx.simple_framebuffer((self.img, self.img), components=4)
        self.prog["fnorm"].value = self.fnorm
        self.prog["nearp"].value = 1.0
        self.prog["scale"].value = self.size
        self.prog["lightdir"].value = (0.4, 0.3, 0.9)

    def render(self, drone_rel, az, el):
        if self.ctx is None:
            self._init_gl()
        fwd, right, up = _basis(az, el)
        self.fbo.use()
        self.ctx.clear(0.0, 0.0, 0.0, 0.0)
        self.prog["cright"].value = tuple(map(float, right))
        self.prog["cup"].value = tuple(map(float, up))
        self.prog["cfwd"].value = tuple(map(float, fwd))
        self.prog["dronepos"].value = tuple(map(float, drone_rel))
        self.vao.render()
        buf = np.frombuffer(self.fbo.read(components=4), np.uint8).reshape(self.img, self.img, 4)
        buf = np.flipud(buf).astype(np.float32) / 255.0   # GL origin is bottom-left
        return buf[:, :, 0], buf[:, :, 3]


def test():
    import detect
    img = 96
    r = DroneRenderer(img=img, size=4.0)

    # projection match: rendered drone centroid must land where detect.project() says
    az, el = 0.25, 0.15
    rel = detect.backproject(0.62, 0.40, 20.0, az, el)   # want it at (u,v)=(0.62,0.40)
    _, u, v, _ = detect.project(rel, az, el)
    gray, mask = r.render(rel, az, el)
    ys, xs = np.nonzero(mask > 0.5)
    assert len(xs) > 4, "drone not rendered / off-frame"
    cu, cv = xs.mean() / img, ys.mean() / img
    assert np.hypot(cu - u, cv - v) < 0.06, f"render vs project mismatch: ({cu:.2f},{cv:.2f}) vs ({u:.2f},{v:.2f})"

    # perspective size falloff: near drone covers more pixels than far
    near = (r.render(detect.backproject(0.5, 0.5, 15.0, az, el), az, el)[1] > 0.5).sum()
    far = (r.render(detect.backproject(0.5, 0.5, 60.0, az, el), az, el)[1] > 0.5).sum()
    assert near > far > 0, f"size should fall with range: near={near} far={far}"
    print(f"render3d checks passed (near={near}px far={far}px, centroid err<0.06)")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test()
    else:
        test()
