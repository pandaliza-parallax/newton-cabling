# Headless recording of Newton's cable-twist example to a rerun .rrd file.
# Three zigzag cables with increasing bend stiffness; the first segment of
# each spins continuously so you can watch twist propagate through the bends.

import newton._src.viewer.viewer_rerun as viewer_rerun_module
from newton.examples.cable.example_cable_twist import Example
from newton.viewer import ViewerRerun

# ViewerRerun calls rr.save() for record_to_rrd but then starts a web/gRPC
# server, which replaces the file sink and leaves the .rrd nearly empty.
# Pretending to be a notebook skips the server launch so the file sink sticks.
viewer_rerun_module.is_jupyter_notebook = lambda: True

NUM_FRAMES = 300  # 5 seconds at 60 fps

viewer = ViewerRerun(
    record_to_rrd="cable_twist.rrd",
    keep_historical_data=True,
)
example = Example(viewer, args=None)

for frame in range(NUM_FRAMES):
    example.step()
    example.render()
    if frame % 60 == 0:
        print(f"frame {frame}/{NUM_FRAMES}", flush=True)

example.test_final()
print("recording complete: cable_twist.rrd (stability checks passed)")
