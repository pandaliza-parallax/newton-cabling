# datagen v2 — LOCKED robot-pose config (single source of truth). DO NOT change these to match a
# code default or a stray command; they must match the REAL RO1 and the old (v1) datagen. Sourced
# by tools/render_batch_v2.sh so the documented values are the ones actually rendered — the doc and
# the run cannot drift apart.
#
# ── Initial hand position (the "initial position of the hand") ─────────────────────────────────
# ARM HOME JOINTS J0..J5 (deg). This is the arm's initial pose AND the IK seed branch (elbow-up).
# CONFIRMED against the physical RO1 teach pendant, Move -> Joints screen (photo, 2026-07-08):
#     J0=4.0  J1=-19.5  J2=-113.0  J3=43.5  J4=-268.9  J5=-178.0
# It equals scripts/record_sbot_scene_gs.py's --arm-home-deg *default* today, but we pass it EXPLICITLY so
# a future edit to that default cannot silently move the hand's start pose (that was the bug: the
# initial pose was only an overridable default, not pinned).
ARM_HOME_DEG="4.0 -19.5 -113.0 43.5 -268.9 -178.0"

# ROBOT BASE in the sbot scene world (m) + base yaw (deg). Locates the whole arm (hence the hand).
# CALIBRATED so the ARM_HOME_DEG gripper starts AT the cable/jack: with the home-relative grasp
# (scripts/record_sbot_scene_gs.py: gripper holds the cable at its HOME orientation and rides the cable's
# relative rotation, instead of slewing to the plug axis), this makes the frame-0 grasp pose land
# within ~4deg of ARM_HOME_DEG and the arm stay near home through the insertion. Derived by shifting
# the v1 base "0.295 -0.45 0.87" by the home-gripper->jack lateral offset [+0.131,+0.344,0].
BASE_POS="0.426 -0.106 0.87"
BASE_YAW_DEG="0"

# GRASP geometry — where the hand holds the plug relative to wrist_3 (defines the grasp pose).
GRASP_RPY="0 -90 0"          # tool-frame orientation of the grasp
GRASP_PROTRUDE="0.015"       # m past the fingertips
GRASP_ALONG_CORD="0.027"     # m back along the cord axis (jaws land behind the plug anchor)

# JACK (socket) world position (m) — the insertion target the trajectory is anchored to.
# z = 0.835: jack CENTER 50mm above the tabletop (0.785). (Briefly 0.815/30mm on 2026-07-30,
# reverted same day.)
JACK_POS="0.295 -0.876 0.835"

# ── v2 jerk fix (see HANDOFF_JERK_ARTIFACT.md) ────────────────────────────────────────────────
GRASPED_ONLY="1"             # record only the already-grasped insertion (no home-hold/approach)
TRIM_SETTLE_MM="3.0"         # drop the physics reset transient at the trajectory start (position)
TRIM_SETTLE_DEG="3.0"        # ... and its orientation tail
START_MIN_OUTSIDE_MM="0"     # if trimming would leave the plug settled at/inside the jack mouth, synthesize a
                             # smooth outside->in approach so it STARTS OUTSIDE (mm of min outside margin; 0=fix inside only)
INSERT_HOLD_HOME_ROT="1"     # gripper HOLDS home orientation during insertion (flexible cord absorbs the
                             # plug's reorientation); arm stays within ~4deg of home the whole insertion.
                             # 0 = gripper rides the cable's rotation instead (can drift 50-70deg on wiggly rollouts)

# ── Cameras / render ───────────────────────────────────────────────────────────────────────────
WRIST_ORBIT="-30"
WRIST_SIDE="-0.06"
WRIST_UP="-0.05"              # wrist-cam eye HEIGHT: lift along world +z (m). Higher = raise cam; negative = below grasp
WRIST_BACK="0.12"           # wrist-cam eye offset BACK along the tool axis (m). Larger = further from the plug
WRIST_AIM_BACK="0.03"        # wrist-cam AIM shift back along the tool axis (m). Larger = tilt view UP toward the gripper fingers
WRIST_CAM_FROM_USD="1"       # 1 = use the REAL eye-in-hand camera from the RO1 USD (/sbot/wrist_3_link/Camera);
                             # exact pose+FOV, ignores the WRIST_ORBIT/SIDE/UP/BACK/AIM_BACK offsets above
# FRONT (main) camera. Placed by angle: ELEV (up/down) + AZIM (around) + DIST_SCALE (distance),
# aimed at the auto scene centre. FRONT_EYE / FRONT_TARGET override that with an exact X Y Z.
DIST_SCALE="0.20"
ELEV="15"
AZIM="-40"
FRONT_TARGET="0.29 -0.87 0.90"
# FRONT_EYE="0.29 -0.4 1.1"    # exact front-cam eye "X Y Z" (m); ~0.5m in front of the gripper. Move y
#                              # toward -0.87 to dolly CLOSER (-0.55~0.35m, -0.65~0.27m); away = farther
# FRONT_TARGET="0.29 -0.87 1.0"  # aim at the gripper/wrist_3 (home ~[0.29,-0.87,1.05])
FRONT_EYE_OFF=""             # cleared (using exact FRONT_EYE/FRONT_TARGET above)

FRONT_TARGET_OFF=""          # cleared

STOP_AFTER_SEAT="20"
# Render resolution. 640x480 = native D405/D415 sensor size, so the calibrated intrinsics
# (--camera-config) are used at their real aspect. NOTE: 4:3, so the square DUMP_SIZE resize
# stretches it -- the real camera images must go through the SAME square resize to stay matched.
# Must equal DalusSimCore renderer._image_width/height (also 640x480).
WIDTH="640"
HEIGHT="480"
DUMP_SIZE="512"              # saved image/wrist_image size (px, square). Was 224. (pi0.5 resizes to 224 at train)
# Global brightness on the rendered RGB (GS has no runtime lights). GAMMA>1 lifts shadows (plug
# interior); GAIN scales overall. Blank = off. Try RENDER_GAMMA="1.8" first.
# ${VAR:-default} so a caller's env override survives this file being sourced
# (a plain assignment silently clobbered per-run RENDER_GAMMA overrides).
RENDER_GAMMA="${RENDER_GAMMA:-1.8}"
RENDER_GAIN="${RENDER_GAIN:-}"
# Third (SIDE +x) camera -> 3-up FRONT|SIDE|WRIST stitch. SIDE_CAM=1 to enable.
SIDE_CAM="1"
SIDE_ELEV=""                 # deg above level (blank = same as ELEV); ignored when SIDE_Z is set
SIDE_Z="0.5"                 # side-cam eye world Z (m): ~0.5m above the table top (0.785)
SIDE_TARGET_Z="0.3"         # side cam aims at jack height (below SIDE_Z) -> ~30deg downward look
# Mirror camera: diagonally OPPOSITE the front cam (front eye reflected through the gripper aim),
# looking back at it -> sees the far side. Preview only. MIRROR_CAM=1 to enable.
MIRROR_CAM="1"
