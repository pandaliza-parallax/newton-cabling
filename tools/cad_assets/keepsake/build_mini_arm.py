"""A cute miniature keepsake: a chubby robot arm holding a heart. ~32 mm tall, prints in one piece."""
import numpy as np
import trimesh
from shapely.geometry import Polygon

OUT_STL = "mini_arm.stl"
OUT_PNG = "mini_arm_preview.png"


def ball(r, center, sub=3):
    s = trimesh.creation.icosphere(subdivisions=sub, radius=r)
    s.apply_translation(center)
    return s


def capsule_between(a, b, r):
    a, b = np.asarray(a, float), np.asarray(b, float)
    v = b - a
    h = np.linalg.norm(v)
    c = trimesh.creation.capsule(height=h, radius=r, count=[24, 24])
    # trimesh capsule axis is +Z spanning z in [-h/2, h/2] (hemispheres beyond)
    T = trimesh.geometry.align_vectors([0, 0, 1], v / h)
    c.apply_transform(T)
    c.apply_translation((a + b) / 2)
    return c


def heart(width=10.0, thickness=4.2):
    t = np.linspace(0, 2 * np.pi, 240)
    x = 16 * np.sin(t) ** 3
    y = 13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)
    poly = Polygon(np.column_stack([x, y])).buffer(0)
    m = trimesh.creation.extrude_polygon(poly, height=thickness)
    m.merge_vertices()
    m.process(validate=True)
    m.fix_normals()
    m.apply_translation(-m.bounding_box.centroid)
    s = width / (m.extents[0])
    m.apply_scale([s, s * 1.1, 1.0])  # slightly taller than wide
    # rotate: extrusion axis z -> y, lobes stay up (+y -> +z)
    m.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    return m


# ---- geometry (mm, z-up) ----
SHOULDER = [0, 0, 6.5]
ELBOW = [0, -4, 17]
WRIST = [0, 3, 24.5]
HEART_C = [0, 11.5, 26.5]

base = trimesh.creation.icosphere(subdivisions=4, radius=14.0)
base.apply_scale([1, 1, 0.42])
base = trimesh.boolean.intersection(
    [base, trimesh.creation.box(extents=[40, 40, 40], transform=trimesh.transformations.translation_matrix([0, 0, 20]))],
    engine="manifold",
)

body_parts = [
    base,
    ball(5.5, SHOULDER),
    capsule_between(SHOULDER, ELBOW, 3.4),
    ball(4.6, ELBOW),
    capsule_between(ELBOW, WRIST, 3.0),
    ball(3.6, WRIST),
    # fingers cradle the heart's flat sides (mirrored in x)
    capsule_between([1.5, 4.0, 25.0], [4.8, 8.5, 26.5], 1.8),
    capsule_between([4.8, 8.5, 26.5], [3.4, 11.5, 27.3], 1.5),
    capsule_between([-1.5, 4.0, 25.0], [-4.8, 8.5, 26.5], 1.8),
    capsule_between([-4.8, 8.5, 26.5], [-3.4, 11.5, 27.3], 1.5),
]

hrt = heart()
hrt.apply_transform(trimesh.transformations.rotation_matrix(np.radians(-14), [1, 0, 0]))
hrt.apply_translation(HEART_C)

body = trimesh.boolean.union(body_parts, engine="manifold")
keepsake = trimesh.boolean.union([body, hrt], engine="manifold")

print("watertight:", keepsake.is_watertight, "| bodies:", keepsake.body_count)
print("extents mm:", [round(x, 1) for x in keepsake.extents])
print("bounds:", [[round(x, 1) for x in b] for b in keepsake.bounds])
print("faces:", len(keepsake.faces))
vol = keepsake.volume / 1000.0
print("volume cm3:", round(vol, 2), "| PLA g (solid):", round(vol * 1.24, 1))

keepsake.export(OUT_STL)

# body/heart as separate meshes for the two-color web viewer
body.export("mini_arm_body.stl")
hrt.export("mini_arm_heart.stl")

# ---- preview render ----
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(10, 5))
for i, (el, az) in enumerate([(12, -78), (18, -20)]):
    ax = fig.add_subplot(1, 2, i + 1, projection="3d")
    for m, col in [(body, "#B6B0A5"), (hrt, "#D96A7B")]:
        ax.plot_trisurf(
            m.vertices[:, 0], m.vertices[:, 1], m.faces, m.vertices[:, 2],
            color=col, edgecolor="none", shade=True,
        )
    ax.set_box_aspect(tuple(keepsake.extents))
    ax.view_init(elev=el, azim=az)
    ax.set_axis_off()
fig.suptitle("mini arm keepsake — 32 mm of devotion")
fig.tight_layout()
fig.savefig(OUT_PNG, dpi=110)
print("wrote", OUT_STL, OUT_PNG)
