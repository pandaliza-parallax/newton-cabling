"""Cut a gripper-finger Gaussian splat into per-link splats using the link meshes.

The finger splat (``sbot_assets/fingers/gripper_fingers.ply``) is a TRELLIS/scan
output in an arbitrary frame & scale, so it must be REGISTERED to the robot's finger
geometry before it can be cut. This project registers splats to CAD the proven way
-- manually in CloudCompare, exporting a 4x4 similarity matrix -- so this tool does
NOT auto-register; it consumes that matrix (or an already-registered splat).

Two steps:

  1) Export a target mesh to align against, then register in CloudCompare:
       python tools/sbot/cut_splat_by_links.py --export-target --theta -0.6
     -> writes fingers_target.obj (the finger links assembled at that opening, in
        the robot/world frame). In CloudCompare: align gripper_fingers.ply onto
        fingers_target.obj (manual rough align + "Register" with scale ON), then
        File > Save the transformation matrix to a 4x4 text file.

  2) Cut the splat using that matrix:
       python tools/sbot/cut_splat_by_links.py --splat .../gripper_fingers.ply \
           --matrix reg.txt --theta -0.6 --out .../sbot_gs
     Each Gaussian is assigned to the nearest finger-link surface, dropped if beyond
     --threshold, transformed into that link's local frame, and written as
     <link>.ply in the flat (SH deg-0) layout your arm splats use, so the link's
     Newton body transform IS its splat transform (1:1, like scripts/record_sbot_gs.py).

IMPORTANT: --theta must match the gripper opening the splat was captured at, so the
assembled meshes line up with the splat. Use --self-test to validate the geometry
end-to-end with no real splat (samples the meshes as a synthetic splat and checks
each point is assigned back to its own link).

Run from the repo root (newton .venv: pxr + trimesh + scipy).
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import newton
import numpy as np
import trimesh
import warp as wp
from pxr import Usd
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from export_gripper_meshes import DEFAULT_USD, _matrix_to_link, _read_mesh

from newton_cabling.sim.sbot import add_sbot, set_gripper

# The eight moving finger links (everything except the static gripper_base_link).
FINGER_LINKS: tuple[str, ...] = (
    "gripper_finger1_knuckle_link",
    "gripper_finger1_inner_knuckle_link",
    "gripper_finger1_finger_link",
    "gripper_finger1_finger_tip_link",
    "gripper_finger2_knuckle_link",
    "gripper_finger2_inner_knuckle_link",
    "gripper_finger2_finger_link",
    "gripper_finger2_finger_tip_link",
)

# The six arm links that become Newton bodies (sbot base_link collapses into the
# world root, gripper_base_link into wrist_3 -- neither has its own body_q).
ARM_LINKS: tuple[str, ...] = (
    "shoulder_link",
    "upper_arm_link",
    "forearm_link",
    "wrist_1_link",
    "wrist_2_link",
    "wrist_3_link",
)

# Selectable link sets. Every entry here is a real Newton body, so its world pose
# comes from FK and the world<->link-local round-trip is exact.
LINK_SETS: dict[str, tuple[str, ...]] = {
    "fingers": FINGER_LINKS,
    "finger1": FINGER_LINKS[:4],   # one finger at a time -> cut each at ITS captured opening
    "finger2": FINGER_LINKS[4:],   # (this scan's two fingers are at different theta)
    "all": ARM_LINKS + FINGER_LINKS,
}

# Flat (SH degree-0) Gaussian layout, matching the arm splats in sbot_gs/flat.
FLAT_PROPS = (
    "x", "y", "z", "scale_0", "scale_1", "scale_2",
    "rot_0", "rot_1", "rot_2", "rot_3", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
)


# ---------------------------------------------------------------- PLY (binary LE)
def read_ply(path: pathlib.Path) -> tuple[list[str], np.ndarray]:
    """Read a binary-little-endian float32 PLY into (property names, N x K array)."""
    raw = path.read_bytes()
    end = raw.find(b"end_header\n") + len(b"end_header\n")
    header = raw[:end].decode("ascii", "replace")
    if "binary_little_endian" not in header:
        raise ValueError("only binary_little_endian PLYs are supported")
    n = next(int(ln.split()[-1]) for ln in header.splitlines() if ln.startswith("element vertex"))
    props = [ln.split()[-1] for ln in header.splitlines() if ln.startswith("property")]
    data = np.frombuffer(raw[end : end + n * len(props) * 4], dtype="<f4").reshape(n, len(props))
    return props, data.astype(np.float64)


def col(props: list[str], data: np.ndarray, name: str) -> np.ndarray:
    return data[:, props.index(name)]


def write_flat_ply(
    path: pathlib.Path,
    xyz: np.ndarray,
    scale: np.ndarray,
    rot_wxyz: np.ndarray,
    f_dc: np.ndarray,
    opacity: np.ndarray,
) -> None:
    n = len(xyz)
    out = np.zeros((n, len(FLAT_PROPS)), dtype="<f4")
    out[:, 0:3] = xyz
    out[:, 3:6] = scale
    out[:, 6:10] = rot_wxyz
    out[:, 10:13] = f_dc
    out[:, 13] = opacity
    header = "ply\nformat binary_little_endian 1.0\n"
    header += f"element vertex {n}\n"
    header += "".join(f"property float {p}\n" for p in FLAT_PROPS)
    header += "end_header\n"
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(out.tobytes())


# ------------------------------------------------------------- similarity (bake)
def _mat_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    x, y, z, w = Rotation.from_matrix(R).as_quat()  # scipy returns xyzw
    return np.array([w, x, y, z])


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product, broadcasting a (4,) over b (N,4); (w,x,y,z) convention."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    return np.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        axis=1,
    )


# --------------------------------------------------------- assemble link meshes
def assemble_links(
    usd_path: pathlib.Path, theta: float, link_names: tuple[str, ...] = FINGER_LINKS
) -> dict[str, dict]:
    """Per link: world-space mesh + (R, t) world transform at gripper ``theta``.

    Link-local geometry comes from the USD (theta-independent); the world pose comes
    from Newton FK with the gripper coupled to ``theta`` -- the same body frames the
    GS render drives, so cutting in this frame round-trips to link-local exactly.
    """
    stage = Usd.Stage.Open(str(usd_path))
    xform_cache = __import__("pxr").UsdGeom.XformCache()
    local_meshes = {}
    for name in link_names:
        link = stage.GetPrimAtPath(f"/sbot/{name}")
        mesh_prim = stage.GetPrimAtPath(f"/sbot/{name}/visuals")
        local_meshes[name] = _read_mesh(
            mesh_prim, _matrix_to_link(mesh_prim, link, xform_cache, world=False)
        )

    builder = newton.ModelBuilder()
    handles = add_sbot(builder, wp.transform(wp.vec3(0, 0, 0), wp.quat_identity()))
    set_gripper(builder, handles, theta)
    model = builder.finalize()
    state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    body_q = state.body_q.numpy()
    labels = list(model.body_label)

    out = {}
    for name in link_names:
        bi = next(i for i, lbl in enumerate(labels) if lbl.endswith(name))
        pos = body_q[bi][:3]
        R = Rotation.from_quat(body_q[bi][3:7]).as_matrix()  # warp quat is xyzw
        world_mesh = local_meshes[name].copy()
        world_mesh.vertices = local_meshes[name].vertices @ R.T + pos
        out[name] = {"mesh": world_mesh, "R": R, "t": pos}
    return out


# --------------------------------------------------------------- assign + cut
def assign_to_links(points: np.ndarray, links: dict[str, dict]) -> tuple[np.ndarray, np.ndarray]:
    """For each point, the index of the nearest link surface and that distance."""
    names = list(links)
    dists = np.empty((len(points), len(names)))
    for j, name in enumerate(names):
        _, d, _ = trimesh.proximity.closest_point(links[name]["mesh"], points)
        dists[:, j] = d
    return dists.argmin(axis=1), dists.min(axis=1)


def cut(
    splat_path: pathlib.Path,
    matrix_path: pathlib.Path | None,
    usd_path: pathlib.Path,
    theta: float,
    threshold: float,
    out_dir: pathlib.Path,
    link_names: tuple[str, ...] = FINGER_LINKS,
    project: bool = False,
    scale_cap: float | None = None,
    iso_scale: float | None = None,
) -> None:
    props, data = read_ply(splat_path)
    xyz = np.column_stack([col(props, data, a) for a in ("x", "y", "z")])
    scale = np.column_stack([col(props, data, f"scale_{i}") for i in range(3)])
    rot = np.column_stack([col(props, data, f"rot_{i}") for i in range(4)])
    rot /= np.linalg.norm(rot, axis=1, keepdims=True)
    f_dc = np.column_stack([col(props, data, f"f_dc_{i}") for i in range(3)])
    opacity = col(props, data, "opacity")

    if matrix_path is not None:
        M = np.loadtxt(matrix_path).reshape(4, 4)
        A, t = M[:3, :3], M[:3, 3]
        s = np.cbrt(abs(np.linalg.det(A)))
        U, _, Vt = np.linalg.svd(A / s)
        R = U @ Vt
        if np.linalg.det(R) < 0:
            R = U @ np.diag([1, 1, -1]) @ Vt
        print(f"registration: scale={s:.5g}  |t|={np.linalg.norm(t):.4g}")
        xyz = s * (xyz @ R.T) + t
        scale = scale + np.log(s)
        rot = _quat_mul(_mat_to_quat_wxyz(R), rot)

    links = assemble_links(usd_path, theta, link_names)
    names = list(links)
    link_idx, dist = assign_to_links(xyz, links)
    keep = dist <= threshold
    print(
        f"{len(xyz)} gaussians; {keep.sum()} within {threshold * 1000:.0f}mm of a link "
        f"({(~keep).sum()} dropped as strays)"
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    for j, name in enumerate(names):
        sel = keep & (link_idx == j)
        if not sel.any():
            print(f"  {name:<40} 0 gaussians (none assigned)")
            continue
        R, t = links[name]["R"], links[name]["t"]
        world_sel = xyz[sel]
        sc = scale[sel]
        note = ""
        if project:  # snap each Gaussian onto the CAD surface -> de-inflate the ~16mm shell
            world_sel, pd, _ = trimesh.proximity.closest_point(links[name]["mesh"], world_sel)
            note = f"  (moved median {np.median(pd) * 1000:.1f}mm onto CAD)"
        if iso_scale is not None:  # round, uniform gaussians (kills anisotropic spikes) -> smooth surface
            sc = np.full_like(sc, np.log(iso_scale))
        elif scale_cap is not None:  # cap linear gaussian size (scale is log-space) -> tight surfels
            sc = np.minimum(sc, np.log(scale_cap))
        local_xyz = (world_sel - t) @ R  # world -> link-local (R^-1 = R.T, applied on right)
        local_rot = _quat_mul(_mat_to_quat_wxyz(R.T), rot[sel])
        short = name.replace("gripper_", "")
        write_flat_ply(
            out_dir / f"{short}.ply", local_xyz, sc, local_rot, f_dc[sel], opacity[sel]
        )
        print(f"  {short + '.ply':<40} {sel.sum():>5} gaussians{note}")


def export_target(
    usd_path: pathlib.Path,
    theta: float,
    out_path: pathlib.Path,
    link_names: tuple[str, ...] = FINGER_LINKS,
) -> None:
    links = assemble_links(usd_path, theta, link_names)
    combined = trimesh.util.concatenate([d["mesh"] for d in links.values()])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.export(out_path)
    ext = combined.extents * 1000.0
    print(
        f"wrote {out_path} ({len(combined.vertices)} verts, "
        f"bbox=[{ext[0]:.0f} {ext[1]:.0f} {ext[2]:.0f}]mm at theta={theta})"
    )
    print(
        "Register gripper_fingers.ply onto this in CloudCompare, save the 4x4, "
        "then re-run with --matrix."
    )


def self_test(
    usd_path: pathlib.Path, theta: float, link_names: tuple[str, ...] = FINGER_LINKS
) -> None:
    """Sample each link mesh, run assignment, report recovery accuracy."""
    links = assemble_links(usd_path, theta, link_names)
    names = list(links)
    pts, truth = [], []
    for j, name in enumerate(names):
        s = links[name]["mesh"].sample(1500)
        pts.append(s)
        truth.append(np.full(len(s), j))
    pts = np.vstack(pts)
    truth = np.concatenate(truth)
    pred, _ = assign_to_links(pts, links)
    acc = (pred == truth).mean()
    print(f"self-test: {len(pts)} sampled points, nearest-link accuracy = {acc * 100:.1f}%")
    for j, name in enumerate(names):
        m = truth == j
        print(f"  {name.replace('gripper_', ''):<34} {(pred[m] == j).mean() * 100:5.1f}% correct")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--splat", type=pathlib.Path, help="finger splat .ply to cut")
    ap.add_argument("--matrix", type=pathlib.Path, help="4x4 splat->robot matrix (CloudCompare)")
    ap.add_argument("--usd", type=pathlib.Path, default=DEFAULT_USD)
    ap.add_argument(
        "--theta", type=float, default=-0.6, help="gripper opening the splat was captured at"
    )
    ap.add_argument(
        "--threshold", type=float, default=0.012, help="max dist (m) to keep a gaussian"
    )
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("gripper_gs"))
    ap.add_argument(
        "--export-target", action="store_true", help="write the assembled finger mesh and exit"
    )
    ap.add_argument(
        "--self-test", action="store_true", help="validate assignment on sampled mesh points"
    )
    ap.add_argument(
        "--links", choices=tuple(LINK_SETS), default="fingers",
        help="which links to cut into: 'fingers' (8) or 'all' (6 arm + 8 finger bodies)",
    )
    ap.add_argument(
        "--project", action="store_true",
        help="snap each kept gaussian onto the CAD surface (de-inflate the scan shell)",
    )
    ap.add_argument(
        "--scale-cap", type=float, default=None,
        help="cap each gaussian's linear size (m), e.g. 0.0015 -> tight surfels",
    )
    ap.add_argument(
        "--iso-scale", type=float, default=None,
        help="force round isotropic gaussians of this size (m), e.g. 0.0025 -> smooth surface, no spikes",
    )
    args = ap.parse_args()
    link_names = LINK_SETS[args.links]

    if args.self_test:
        self_test(args.usd, args.theta, link_names)
        return
    if args.export_target:
        export_target(args.usd, args.theta, args.out / f"{args.links}_target.obj", link_names)
        return
    if not args.splat:
        ap.error("provide --splat (and usually --matrix), or use --export-target / --self-test")
    cut(args.splat, args.matrix, args.usd, args.theta, args.threshold, args.out, link_names,
        project=args.project, scale_cap=args.scale_cap, iso_scale=args.iso_scale)


if __name__ == "__main__":
    main()
