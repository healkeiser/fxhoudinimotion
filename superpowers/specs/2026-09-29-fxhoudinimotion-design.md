# fxhoudinimotion: Kimodo, ARDY and MotionBricks in Houdini

Date: 2026-09-29
Status: draft, awaiting review

## 1. Goal

Turn the repository (renamed from fxhoudinikimodo to fxhoudinimotion on
2026-09-29) into a Houdini toolset for character animation authoring built on
three NVIDIA motion models:

| Model | Strength | Skeleton | Controls | Native fps | Live |
|---|---|---|---|---|---|
| Kimodo | Offline, high quality authoring | SOMA77 (human) | Text timeline, root path, poses | 30 | No |
| ARDY | Real time, text + constraints | Core (human), G1; SOMA announced | Streamed text, keyframes, root path, hands/feet | 20 (Core), 25 (G1) | Yes |
| MotionBricks (public release) | Real time styled locomotion | G1 (robot) only | Style + movement/facing direction | 30 | Yes |

The models coexist as separate generator nodes on a shared foundation, and a
mixing toolset puts their output on one skeleton, sequences it, layers it by
body part and moves it onto the user's own rig. ARDY and MotionBricks can also
be driven live from a gamepad, with the take recorded and baked to an ordinary
clip.

Primary use: character animation authored in Houdini (VFX / film). Robotics
and game runtime are out of scope.

### Facts established before this spec

- MotionBricks runs headless on native Windows (Python 3.11, torch 2.6 cu124,
  RTX 4090): models load in 2.6 s, 13 s of motion is generated in 2.2 s. It
  needs the `keyboard` package, which its `setup.py` omits. Its keyboard
  controllers can be bypassed by building the control dict directly
  (`movement_direction`, `facing_direction`, `mode`,
  `allowed_pred_num_tokens`, `context_mujoco_qpos`). Throwaway spike:
  `scripts/motionbricks_spike.py`, `scripts/motionbricks_load_sop.py`.
- The MotionBricks human model shown in the paper's UE5 demos is not released;
  the paper retargets it live in UE5. The public weights are G1 only.
- ARDY ships `scripts/generate.py` (text to NPZ with `posed_joints`,
  rotations, root, foot contacts). It uses the same gated
  Meta-Llama-3-8B-Instruct (LLM2Vec) text encoder as Kimodo. Windows is
  untested upstream.

## 2. Architecture

```
 GPU side: one process per model, one HTTP/WS contract        Houdini
 +----------------+ +--------------+ +-------------------+   +------------------------------+
 | kimodo server  | | ardy server  | | motionbricks srv  |   | fxmotion (Python library)    |
 | Docker (today) | | own env      | | own venv          |<--+  client, clip loader,        |
 +-------+--------+ +------+-------+ +---------+---------+   |  skeleton registry, timeline |
         +-- /health /generate /jobs /download /cancel /live -> +--------------+---------------+
               every clip comes back in the fxmotion.clip/1 format        |
                                   generators: kimodo_motion 2.0, ardy_motion, motionbricks_motion
                                   mixing:     motion_retarget, motion_sequence, motion_layer
```

Decisions:

1. One process per model. The three have incompatible dependency sets (Kimodo
   lives in its Linux Docker image, MotionBricks pulls transformers 5 and
   mujoco, ARDY is unknown). What they share is a server module (jobs, cache,
   progress, cancel, live sessions) and the clip format.
2. Each process loads its model on the first request and unloads it after an
   idle timeout (`FXMOTION_IDLE_UNLOAD_S`), so VRAM holds one or two models at
   a time next to Houdini.
3. Conversion to the common format happens on the server, in a per-model
   adapter. Houdini has exactly one clip loader.
4. The mixing toolset is thin HDAs over native KineFX nodes, never custom
   solvers.
5. Live mode is a way of authoring a clip, not a second pipeline: a take is
   baked to the same format and flows through the same nodes.

## 3. Clip format: `fxmotion.clip/1`

An NPZ. Required keys:

| Key | Shape / type | Meaning |
|---|---|---|
| `format` | str | `"fxmotion.clip/1"` |
| `skeleton` | str | `"soma77"`, `"ardy_core"` or `"g1"` |
| `joint_names` | (J,) str | |
| `parents` | (J,) int32 | parent index, -1 for the root |
| `fps` | float32 | the model's native rate |
| `world_pos` | (T, J, 3) float32 | Houdini space: Y up, metres |
| `world_rot` | (T, J, 3, 3) float32 | Houdini row-vector convention, ready for the `transform` attribute |
| `rest_pos` | (J, 3) float32 | rest pose, for Biped Setup |
| `rest_rot` | (J, 3, 3) float32 | |

Optional keys:

| Key | Shape / type | Meaning |
|---|---|---|
| `contacts` | (T, J) int8 | foot contacts where the model predicts them |
| `segments` | JSON str | per segment: `prompt` or `style`, `start`, `end` (frames) |
| `source` | JSON str | backend, model id, request hash, seed |
| `native_*` | any | the model's raw arrays, kept for model-specific features (Kimodo Continue / Regenerate use `native_local_rot_mats`, `native_root_positions`) |

Invariants (enforced by tests, section 8): rotations orthonormal, parents form
a tree rooted at one joint, lowest foot joint within 5 cm of y = 0 at rest.

## 4. Server contract

### Batch

- `GET /health` returns `{backend, models, skeleton, fps, capabilities}`,
  where `capabilities` is a set of `text`, `segments`, `root_path`,
  `keyframes`, `effectors`, `styles` (list of style names), `live`. The
  Houdini nodes read it to hide the controls a model does not support.
- `POST /generate` takes one envelope for every backend:
  - `segments`: `[{duration_s, prompt? , style?}]`
  - `root_path`: points in Houdini space, with times in seconds
  - `keyframes`: `[{time_s, joint_names, world_pos, world_rot}]`
  - `effectors`: `[{time_s, joint, world_pos?, world_rot?}]`
  - `continue_from`: the tail of a clip in `fxmotion.clip/1`
  - `seed`, `model`, `force`
  Times are seconds and positions are Houdini space, so frame rates stay on
  the server. A control outside the backend's capabilities returns 422 with
  a message naming both (`"motionbricks: keyframes not supported"`); nothing
  is ignored silently.
- `GET /jobs/{id}`, `GET /jobs/{id}/download`, `POST /jobs/{id}/cancel`: as
  today, including the request cache and progress reporting.

### Live: `WS /live`

- Client to server: `start {model, seed, initial_pose?}`, then per tick
  `control {move: [x, z], face: [x, z], style? , prompt?}`, then `stop` or
  `bake`.
- Server to client: `frames {index, world_pos, world_rot}` as float32 buffers.
  The server owns the session's motion history and generates a short horizon
  ahead (MotionBricks replans every 8 frames).
- `bake` writes the session as `fxmotion.clip/1` and returns a job id whose
  `/download` serves it, so a live take arrives through the batch path.
- If the socket drops mid-take, the frames Houdini already received are kept
  and can be baked on the Houdini side.

## 5. Houdini side

### `fxmotion` library (`houdini/python/fxmotion/`)

- `client`: HTTP jobs and the WebSocket session, shared by every node.
- `clip`: `fxmotion.clip/1` to KineFX geometry; the only loader.
- `skeletons/`: one entry per skeleton (`soma77`, `ardy_core`, `g1`) with its
  rest pose, its rest geometry if any, and its Biped Setup mapping as JSON.
- `timeline`: the current `kimodo_timeline` panel, made generic; a segment
  holds a `prompt` or a `style` depending on the backend's capabilities.

Every generator output carries detail attributes `fxmotion_skeleton`,
`fxmotion_fps` and `fxmotion_source`. Mixing nodes read them to configure
themselves and to fail with a clear message on a mismatch.

### Generators

All keep the current four outputs: 0 Rest Geometry, 1 Capture Pose,
2 Animated Pose, 3 T-Pose. Shared parameters: Server, Model, Seed, Start
Frame, Retime to Scene FPS, Generate, Cancel, status under the node.

| Node | Controls | Live |
|---|---|---|
| `vb::kimodo_motion::2.0` | as 1.1: prompt timeline, root path, poses | No |
| `vb::ardy_motion::1.0` | prompt timeline, root path (input 0), keyframes (input 1), hands/feet | Yes |
| `vb::motionbricks_motion::1.0` | style timeline, root path from a curve (input 0, tangent to direction) | Yes |

- `vb::kimodo_motion::1.1` stays in the library so saved scenes load; new
  nodes are 2.0. This is what allows the internals to be rewritten.
- Live tab (ARDY, MotionBricks): Create Gamepad button (builds a Gamepad CHOP
  beside the node and wires it), an editable control mapping, a
  camera-relative toggle, Record, Stop & Bake. Default mapping: left stick is
  the travel direction relative to the viewport camera, right stick the
  facing direction, buttons and d-pad pick MotionBricks styles or ARDY preset
  prompts.
- Rest geometry: SOMA77 uses the current skin. G1 uses the MotionBricks STL
  meshes, each rigidly bound to its joint, subject to the licence check in
  section 9. ARDY Core uses its mesh if ARDY provides one, otherwise output 0
  is empty and the node warns.

### Mixing toolset

| HDA | Inputs | Inside |
|---|---|---|
| `vb::motion_retarget::1.0` | 0: any generator clip; 1 (optional): the user's rig. Without input 1 the target is SOMA77 | Rig Stash Pose + Biped Setup on each side (source mapping from the registry, keyed by `fxmotion_skeleton`), Biped Retarget |
| `vb::motion_sequence::1.0` | A, B, same skeleton | MotionClip on each input, MotionClip Sequence (Compute Locomotion on the root joint, Match Translation, orientation Around Up Axis, Blend Frames), MotionClip Evaluate |
| `vb::motion_layer::1.0` | base, overlay, same skeleton | Skeleton Blend in local space, Group from the Biped Setup groups (`biped_upperbody`, `biped_leg`, ...), Bias as the weight |

`motion_retarget` rewrites `fxmotion_skeleton` on its output to the target
(`soma77`, or `custom` for a user rig) so the nodes downstream can check it.
Two `custom` clips are treated as the same skeleton only when their
`joint_names` match.

Typical chain: `motionbricks_motion` then `motion_retarget`, and `ardy_motion`,
into `motion_layer` (legs and root from MotionBricks, upper body from ARDY),
then `motion_retarget` onto the user's rig, then Joint Deform.

All nodes sit in a "fxmotion" TAB submenu. No custom shelf.

## 6. Renames

- `houdini/python/kimodo_timeline` becomes `fxmotion.timeline`.
- `KIMODO_BRIDGE_ROOT` becomes `FXMOTION_ROOT`. The old name is not read.
- The HDA namespace stays `vb::`.
- `kimodo_server.py` becomes the Kimodo adapter on the shared server module.
- The local clone folder is renamed only if the user decides to; it changes
  the Houdini package path.

## 7. Phasing

### Phase 0: spikes (throwaway code, each answers go / no-go)

1. ARDY on Windows: does `generate.py` run with Core; does batch generation
   accept root path and keyframes or only the demo loop; what is the Core
   joint layout; is a mesh provided.
2. Live loop: MotionBricks over WebSocket, Gamepad CHOP, a live SOP. Measure
   round-trip latency and viewport pacing, playbar playback against an
   event-loop callback.
3. G1 to SOMA77 retarget: finish the Biped Setup / Biped Retarget network
   already validated as a dry run in `/obj/mb_spike`, and judge the result on
   the walk / zombie / dance spike clip.

If spike 1 fails on Windows, ARDY runs in a Linux container like Kimodo;
nothing upstream changes.

### Phases 1 to 6 (each usable on its own)

| Phase | Deliverable | Done when |
|---|---|---|
| 1 | Clip format, `fxmotion` library, shared server module, Kimodo on top (`kimodo_motion` 2.0) | 2.0 matches 1.1 numerically on a reference clip (positions within 1e-5) |
| 2 | `motion_retarget` with `soma77` and `g1` mappings | G1 to SOMA77 and SOMA77 to a Houdini Test Geometry character cook without errors, feet on the ground |
| 3 | `motionbricks_motion`, batch | style timeline and curve produce a clip in Houdini |
| 4 | `ardy_motion`, batch | prompt, root path and keyframes, as far as spike 1 allows |
| 5 | Live: MotionBricks, then ARDY | a baked take matches the frames shown live |
| 6 | `motion_sequence`, `motion_layer` | A then B with no pop at the seam; layering respects the mask |

MotionBricks precedes ARDY because it is proven on Windows.

## 8. Testing

- Offline, in CI (pytest):
  - clip-format invariants for every adapter, on small real clips from each
    model kept as fixtures;
  - server contract in `MOCK_MODE`: 422 on unsupported controls, the live
    session protocol, bake through `/download`.
- Live Houdini (`tests/test_houdini_live.py`): each node cooks, has its four
  outputs and its `fxmotion_*` attributes; mixing nodes refuse mismatched
  skeletons.
- Regression: `kimodo_motion` 1.1 against 2.0 (phase 1).

## 9. Licences and risks

- `LICENSE` and the README list the three upstreams: Kimodo's terms; ARDY and
  MotionBricks, Apache-2.0 code and NVIDIA Open Model weights; the gated
  Llama licence of the text encoder.
- The G1 meshes are Unitree's; their licence must be checked before they are
  redistributed in this repository. Until then the node loads them from the
  user's MotionBricks checkout.
- Moving targets: ARDY's SOMA model and a human MotionBricks model are
  announced, not released. The skeleton registry is the one place a new
  skeleton is added.
- VRAM: Kimodo (3 to 4 GB) and ARDY's text encoder (about 14 GB in bf16) do
  not fit together next to Houdini on 24 GB; the idle unload in section 2 and
  a CPU text encoder option address it.
- Live pacing in Houdini is unmeasured; spike 2 decides the mechanism.
