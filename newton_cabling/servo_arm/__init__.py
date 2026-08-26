"""Hardware-calibrated servo-arm plant bank for the cable-insertion envs.

``ServoArmBank`` turns per-env joint GOALS into realistic joint TRAJECTORIES
using the RO2-core calibration from the sibling parallax-demo-newton repo
(``sysid/`` + ``control/``). Pure CPU: numpy + mujoco + ruckig/toppra — no
torch, no warp, no newton — so it imports and tests in the no-GPU gate.

See newton_cabling/servo_arm/bank.py for the full rationale and the frozen
contract this implements (newton_cabling/servo_arm/CONTRACT.md).
"""

from newton_cabling.servo_arm.bank import ServoArmBank

__all__ = ["ServoArmBank"]
