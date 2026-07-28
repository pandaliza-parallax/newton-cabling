# Headless, scripted run of Newton's RJ45 plug/socket insertion example,
# recorded to a rerun .rrd file.
#
# The interactive example drives the plug with a mouse gizmo; here we drive
# the same spring target programmatically through a timeline:
#   0.0-1.5s  settle (cable sags under gravity)
#   1.5-4.5s  slow insertion along +Y until the latch clicks in
#   4.5-6.0s  hold
#   6.0-8.0s  pull-back test: drag the target well past the rest pose --
#             if the latch mechanism works, the plug stays seated
#   8.0-9.0s  release

import newton._src.viewer.viewer_rerun as viewer_rerun_module
import warp as wp
from newton.examples.contacts.example_contacts_rj45_plug import Example
from newton.viewer import ViewerRerun

# ViewerRerun calls rr.save() for record_to_rrd but then starts a web/gRPC
# server, which replaces the file sink and leaves the .rrd nearly empty.
# Pretending to be a notebook skips the server launch so the file sink sticks.
viewer_rerun_module.is_jupyter_notebook = lambda: True

FPS = 60
DURATION_SECONDS = 9.0
INSERT_DEPTH = 0.035  # meters past rest pose (plug starts 25mm out of socket)
PULLBACK_DEPTH = -0.05


def target_offset_y(t: float) -> float:
    if t < 1.5:
        return 0.0
    if t < 4.5:
        return INSERT_DEPTH * (t - 1.5) / 3.0
    if t < 6.0:
        return INSERT_DEPTH
    if t < 8.0:
        return INSERT_DEPTH + (PULLBACK_DEPTH - INSERT_DEPTH) * (t - 6.0) / 2.0
    return PULLBACK_DEPTH


viewer = ViewerRerun(
    record_to_rrd="rj45_insertion.rrd",
    keep_historical_data=True,
)
example = Example(viewer, args=None)

rest = example._rest_pos
num_frames = int(DURATION_SECONDS * FPS)

for frame in range(num_frames):
    t = example.sim_time
    target = wp.vec3(rest[0], rest[1] + target_offset_y(t), rest[2])

    # Spring mode (-1 = no body picked), aim the plug spring at the target.
    example._pick_body.assign([-1])
    example._pick_target.assign([target])
    example.gizmo_tf = wp.transform(target, wp.quat_identity())

    if example.graph:
        wp.capture_launch(example.graph)
    else:
        example.simulate()
    example.sim_time += example.frame_dt
    example.render()

    if frame % 60 == 0:
        plug_y = float(example.state_0.body_q.numpy()[example._plug_body][1])
        print(
            f"t={t:4.1f}s target_dy={target_offset_y(t):+.3f} plug_y={plug_y:+.4f}",
            flush=True,
        )

final_plug_y = float(example.state_0.body_q.numpy()[example._plug_body][1])
seated = abs(final_plug_y - (rest[1] + 0.025)) < 0.01
print(f"final plug_y={final_plug_y:+.4f} rest_y={rest[1]:+.4f} latch_held={seated}")
print("recording complete: rj45_insertion.rrd")
