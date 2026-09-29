# Phase 0: spike findings (2026-09-29)

Results of the three spikes in section 7 of
`2026-09-29-fxhoudinimotion-design.md`. The spike code was throwaway (it lived
in the session scratchpad and in git-ignored `scripts/motionbricks_*.py`);
these are the answers it produced, for planning phases 2 to 5.

## Spike 3: G1 (MotionBricks) to SOMA77 retarget - partial

Setup: MotionBricks clip loaded as a KineFX skeleton (MuJoCo body frames, Z up
to Y up), Rig Stash Pose + Biped Setup on each side, Biped Retarget onto the
SOMA77 skeleton of `kimodo_motion` 2.0, Joint Deform.

Works, measured over a 13 s walk / zombie / stealth / dance clip:

- root path follows the G1's, strides scaled about 1.25x to the taller human
  (expected from Biped Retarget's size match);
- feet on the ground throughout (lowest foot -0.016 to 0.029 m);
- no pops (largest joint step 0.18 m per frame, during the dance);
- the T-pose pairing is exact: the G1 fed its own T-pose gives the SOMA77
  T-pose within 0.2 degrees.

Does not work: the arms. On the walk the G1's arms hang about 30 degrees from
vertical with 30 degrees of elbow bend; the human gets either 28 / 50 degrees
(rest pose = G1 qpos0) or about 85 / 80 degrees (with a straightened T-pose,
through Biped Setup's Pull or a true T-pose rest computed in MuJoCo).

Facts learned on the way:

- SOMA77 is not recognised by any Biped Setup preset; it needs a Custom
  mapping (Hips, Chest, Neck1, Head, LeftArm, LeftHand, LeftLeg, LeftFoot,
  LeftToeBase and the right side). This belongs in the skeleton registry.
- Biped Setup's Mapping menu is integer-indexed (1 = Custom).
- The G1's zero pose has the elbows bent 82 degrees; a straight arm is
  elbow +82 degrees, a horizontal one shoulder roll +/-89 degrees.

Untested hypothesis for the arms: each G1 shoulder, hip and wrist is a chain of
three hinge bodies at nearly the same point, where the human has one joint.
Collapsing the G1 to a human-like skeleton (one joint per shoulder, hip and
wrist) before Biped Setup should give the IK a comparable chain. This belongs
in the MotionBricks adapter (phase 3), not in the retarget HDA.

## Spike 1: ARDY on Windows - yes

- Runs natively on Windows 11 in a uv venv (Python 3.11, torch 2.6 cu124).
  `pip install -e .` builds the C++ MotionCorrection extension with CMake 4.1
  and Visual Studio 2022; nothing else was needed.
- 5 s of motion with a root-path constraint: 11 s end to end on an RTX 4090
  (the first run adds about 30 s of checkpoint download).
- Batch generation (`scripts/generate.py`) takes `--constraints`, a JSON list
  with the same constraint types and fields as Kimodo (`root2d` with
  `frame_indices` + `smooth_root_2d`, `fullbody`, `end-effector`). A four
  waypoint path was followed within 3 cm.
- The text encoder is the same LLM2Vec / Llama 3 8B as Kimodo's, with the same
  client (`/DemoWrapper` on port 9550): ARDY used the running Kimodo
  text-encoder container (`TEXT_ENCODER_MODE=api`), so the two models share one
  14 GB encoder.
- Core skeleton: 27 joints with conventional names (Hips, Spine..Spine3, Neck,
  Head, Left/RightShoulder, Arm, ForeArm, Hand, HandEnd, HandThumb1, UpLeg,
  Leg, Foot, ToeBase), 20 fps, parent-first order. A skin is shipped
  (`ardy/assets/skeletons/cskel27/skin_standard.npz`, Kimodo's format).
- The output NPZ has exactly Kimodo's keys (`posed_joints`, `global_rot_mats`,
  `local_rot_mats`, `root_positions`, `smooth_root_pos`, `foot_contacts` with 4
  channels, `global_root_heading`) in the same canonical space. The Kimodo
  adapter generalises: the skeleton (`cskel27`, T-pose offsets to compute), the
  fps and the contact channels are what change.

Consequence: ARDY Core is human, so the G1 retarget (spike 3) matters much less
for human animation.

## Spike 2: live gamepad loop - yes, with the playbar

Setup: MotionBricks agent behind a plain HTTP server (one `/step` per frame,
keep-alive), Gamepad CHOP, a Python SOP drawing the returned pose.

- The Gamepad CHOP sees an XInput controller out of the box. It reads stick up
  as `lsy = -1` (screen convention).
- Camera-relative control works with the viewport's own camera
  (`curViewport().viewTransform()`, camera-to-world, row-vector): stick
  forward = away from the view, projected on the ground. No camera object is
  needed. When the view looks straight down the projected forward degenerates;
  use the camera's up axis then.
- Round trip (HTTP loopback + one generated frame): 2.4 ms median; SOP rebuild
  0.7 ms.
- Pacing:
  - an event-loop callback (`hou.ui.addEventLoopCallback`) only runs about
    every 55 ms, so about 18 fps whatever the work costs;
  - real-time playbar playback, with the SOP pulling a step per cooked frame,
    runs at the scene rate (39 ms median at 24 fps) and felt "a lot more
    reactive". Use the playbar.
- A Python SOP whose code reads no parm is not dirtied by a parm change; the
  live SOP must be time-dependent (reference the frame) or force-cooked.
- MotionBricks replans every 16 frames and a replan costs 70 to 95 ms, done
  synchronously in `/step`: a visible hitch twice a second. The live server
  must generate ahead in a background thread so a step never waits on a
  replan.
- Playbar pacing ties the step rate to the scene fps (24 against the model's
  30): the live SOP has to step by clip time, not once per frame.
