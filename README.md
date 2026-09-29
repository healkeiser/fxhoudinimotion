<div align="center">

  <p align="center">
    <img src="https://raw.githubusercontent.com/healkeiser/fxhoudinimotion/main/docs/images/banner.webp" alt="fxhoudinimotion: NVIDIA motion models in SideFX Houdini" width="100%">
  </p>

  <p align="center">
    <!-- Maintenance status -->
    <img src="https://img.shields.io/badge/maintenance-actively--developed-brightgreen.svg?&label=Maintenance">&nbsp;&nbsp;
    <!-- License -->
    <img src="https://img.shields.io/badge/License-MIT%20%2B%20upstream%20terms-blue.svg?&logoColor=white" alt="License: MIT + upstream terms"/>&nbsp;&nbsp;
    <!-- Last Commit -->
    <img src="https://img.shields.io/github/last-commit/healkeiser/fxhoudinimotion?logo=github&label=Last%20Commit" alt="Last Commit"/>&nbsp;&nbsp;
    <!-- Commit Activity -->
    <a href="https://github.com/healkeiser/fxhoudinimotion/pulse" alt="Activity">
      <img src="https://img.shields.io/github/commit-activity/m/healkeiser/fxhoudinimotion?&logo=github&label=Commit%20Activity"/></a>&nbsp;&nbsp;
    <!-- GitHub stars -->
    <img src="https://img.shields.io/github/stars/healkeiser/fxhoudinimotion" alt="GitHub Stars"/>
  </p>

</div>

<!-- TABLE OF CONTENTS -->
## Table of Contents

- [About](#about)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Writing Prompts](#writing-prompts)
- [Regenerating Part of a Clip](#regenerating-part-of-a-clip)
- [Environment Variables](#environment-variables)
- [Development](#development)
- [Contributing](#contributing)
- [Credits](#credits)
- [Contact](#contact)
- [License](#license)

<!-- ABOUT -->
## About

[Kimodo](https://github.com/nv-tlabs/kimodo) is NVIDIA Research's kinematic motion diffusion model: give it an English sentence such as `a person walks forward slowly`, optionally a root path or pose keyframes, and it generates a 77-joint SOMA human motion clip. It runs locally on an NVIDIA GPU.

**fxhoudinimotion** brings it into Houdini as a single SOP, `kimodo_motion`. The node talks to a small FastAPI server that keeps the model resident on the GPU, pulls the result back over HTTP, and rebuilds it as KineFX geometry: a skinned body, its capture pose, the animated skeleton and a T-pose, in the same output order as Houdini's own Test Geometry characters. A Joint Deform wired straight across gives you a moving body; Biped Setup and Biped Retarget move that motion onto your own rig.

This is a fork of [chordee/kimodo-houdini-bridge](https://github.com/chordee/kimodo-houdini-bridge) with a reworked node interface, scene-FPS retiming and Windows fixes. See [Credits](#credits).

<!-- FEATURES -->
## Features

| Area | What you get |
|------|--------------|
| **Prompt to motion** | Multi-line prompt, duration in scene frames, model choice, one Generate button. Runs in the background; the node recooks when the clip lands. |
| **Timeline panel** | Dockable Python Panel: prompt segments laid end to end (drag to resize or reorder, double-click to edit, split at playhead), Kimodo transition length, and Full Body / hand / foot pose tracks with draggable keys. Everything is stored on the node and undoable. |
| **SideFX output order** | 0 Rest Geometry, 1 Capture Pose, 2 Animated Pose, 3 T-Pose. `kinefx::jointdeform` wires 0 → 0, 1 → 1, 2 → 2. |
| **Timing that matches your scene** | Kimodo samples at 30 fps. Start Frame (default `$FSTART`) and Retime to Scene FPS keep a 3 s clip at 3 s whether you work at 24, 25 or 30. |
| **Constraints** | Root path from any curve or points on input 0. Full-body or hand/foot pose keyframes from a posed skeleton on input 1, with a Create Pose Rig button. Raw Kimodo constraint JSON if you prefer. |
| **Foot contacts** | Kimodo's per-frame contact labels arrive as an `int contact` point attribute, ready for foot locking or footstep FX. |
| **Server that stays warm** | Model preloaded once, results cached by prompt + duration + model + constraints, text encoder on CPU to spare VRAM. Runs on your workstation or a remote GPU box. |
| **Status where you look** | Job state shown under the node in the network editor, plus a Test Connection button. No modal dialogs on Generate. |
| **Offline loader** | Point Clip Path at any `fxmotion.clip/1` file and the node rebuilds it without a server. |
| **ARDY Motion** | A second generator on NVIDIA ARDY's 27-joint Core skeleton: same timeline, root path and pose keys, 12 s of motion in about 4 s on an RTX 4090, sharing Kimodo's text encoder. |

<!-- ARCHITECTURE -->
## Architecture

```mermaid
flowchart LR
    subgraph Houdini[" 🔶 SideFX Houdini "]
        direction TB
        H1("kimodo_motion SOP")
        H2("kinefx::jointdeform")
        H3("Biped Setup / Retarget")
        H1 --> H2
        H1 --> H3
    end

    subgraph Server[" ⚡ kimodo_backend · Docker "]
        direction TB
        S1("FastAPI · port 8001")
        S2("Kimodo diffusion · GPU")
        S3("LLM2Vec text encoder · CPU · port 9550")
        S1 --> S2
        S2 --> S3
    end

    H1 -. "POST /generate · GET /jobs/{id}/download" .-> S1

    classDef houdiniBox fill:#fff5f0,stroke:#e8c4a8,stroke-width:1px,color:#2d3748,rx:12,ry:12
    classDef serverBox fill:#eef6f0,stroke:#a8d5b8,stroke-width:1px,color:#2d3748,rx:12,ry:12
    classDef houdiniNode fill:#fde4d0,stroke:#e0a87c,stroke-width:1px,color:#2d3748,rx:8,ry:8
    classDef serverNode fill:#d4edda,stroke:#82c896,stroke-width:1px,color:#2d3748,rx:8,ry:8

    class Houdini houdiniBox
    class Server serverBox
    class H1,H2,H3 houdiniNode
    class S1,S2,S3 serverNode
```

Diffusion needs about 4 GB of VRAM and stays resident while the `api` container runs. Text encoding is offloaded to CPU. Houdini only ever sees HTTP and an NPZ file, so the server can live on any machine you can reach.

<!-- INSTALLATION -->
## Installation

Two halves: a **Docker server** that runs Kimodo, and a **Houdini package** that loads the HDA. The full walkthrough, including Windows specifics, is in [docs/setup.md](docs/setup.md).

### Requirements

- **Houdini** 20.5+ (developed on 22.0.368)
- **Docker Desktop** with the WSL2 backend and GPU support on Windows, or Docker Engine + NVIDIA Container Toolkit on Linux
- **NVIDIA GPU**. How much VRAM and RAM you need depends on where you run the text encoder, which is the largest thing Kimodo loads. Pick one before installing, see [Where to run the text encoder](#where-to-run-the-text-encoder):
  - encoder on **CPU** (the default): about 4 GB VRAM, 6 GB if Houdini shares the card, but **~14 GB of system RAM**, and text encoding becomes the slowest part of every generation
  - encoder on **GPU**: about **17 GB VRAM**, plus whatever your viewport needs, and generation is much faster
- **RAM**: 16 GB is enough with the encoder on GPU; budget 32 GB with it on CPU
- **~50 GB disk**: the CUDA base image alone is 35 GB, plus model weights and the Hugging Face cache
- A **Hugging Face** account. The text encoder is built on Meta's Llama 3 8B Instruct, which is gated; see the setup guide for the mirror route if Meta declines your request.

### Server

Clone [healkeiser/kimodo](https://github.com/healkeiser/kimodo) rather than NVIDIA's repository directly. It is a fork of `nv-tlabs/kimodo` that is one commit ahead: `initial_motion` on `_multiprompt`, which [regenerating part of a clip](#regenerating-part-of-a-clip) needs. The commit is purely additive, so everything else behaves identically, and upstream is still there as a remote to pull from.

```shell
git clone https://github.com/healkeiser/kimodo.git
cd kimodo
git remote add upstream https://github.com/nv-tlabs/kimodo.git
git clone https://github.com/nv-tlabs/kimodo-viser.git
docker build -t kimodo:1.0 .

# the compose file goes into the kimodo dir; the server code is mounted from
# this repo, so point FXMOTION_ROOT at it (in .env on Windows, see below)
cp /path/to/fxhoudinimotion/docker-compose.bridge.yaml .
export FXMOTION_ROOT=/path/to/fxhoudinimotion
mkdir -p output

# weights (Kimodo is ungated; the Llama-based text encoder needs your HF token)
hf download nvidia/Kimodo-SOMA-RP-v1.1

docker compose -f docker-compose.bridge.yaml up text-encoder -d   # wait for "healthy"
MOCK_MODE=0 docker compose -f docker-compose.bridge.yaml up api -d
curl http://localhost:8001/health     # {"backend":"kimodo", ..., "mock_mode":false}
```

On Windows, add a `.env` next to the compose file with `HF_HOME=C:/Users/<you>/.cache/huggingface` and `FXMOTION_ROOT=C:/Users/<you>/Documents/GitHub/fxhoudinimotion`; the compose file falls back to `$HOME`, which Windows does not set. Large weight downloads are faster on the host (`uv tool run --from huggingface_hub hf download ...`) than through the Docker bind mount.

### ARDY server (optional)

[NVIDIA ARDY](https://github.com/nv-tlabs/ardy) generates the same kind of motion on its own 27-joint Core skeleton, with the same timeline, root path and pose keys, through the **ARDY Motion** node. It runs natively in its own Python environment (no Docker) and borrows Kimodo's text-encoder container, so the Kimodo server above must be set up first; only its `text-encoder` service needs to run.

```shell
git clone https://github.com/nv-tlabs/ardy.git   # next to this repo
cd ardy
# create ARDY's .venv as its README says (Python 3.11, CUDA torch, `-e .`,
# which builds a C++ extension: CMake + Visual Studio 2022 on Windows), then:
uv pip install --python .venv fastapi uvicorn requests

docker compose -f docker-compose.bridge.yaml up text-encoder -d   # from the kimodo dir
scripts\run_ardy_server.ps1                                      # from this repo, port 8002
curl http://localhost:8002/health     # {"backend":"ardy", ...}
```

The first start downloads the ARDY checkpoint (about 30 s). Stop the Kimodo `api` container when you only use ARDY: the text encoder is shared, the motion models are not.

### Houdini

The node uses `requests` and `numpy`, which ship with Houdini. The panel needs
[QtPy](https://pypi.org/project/QtPy/), which does not, so put it in `vendor/`, which the
package file adds to `PYTHONPATH`. Never `pip install` into Houdini's Python or your user
site-packages.

```shell
python -m pip install --target vendor --no-deps QtPy
```

QtPy is a pure-Python wheel, so any Python 3.7+ can install it; `--no-deps` is safe because
its only runtime dependency, `packaging`, already ships with Houdini.

Copy `fxhoudinimotion.json` into `$HOUDINI_USER_PREF_DIR/packages/` and set `FXMOTION_ROOT` in it to this repo's absolute path. Restart Houdini. The node appears under **Tab ▸ fxmotion**.

Scenes saved with the older **Kimodo Motion 1.1** node still load and still play the clips they already downloaded, but 1.1 cannot Generate against the current server. Drop a new node (2.0) to generate.

<!-- USAGE -->
## Usage

1. Drop a **Kimodo Motion** SOP. Press **Test Connection** on the Server tab; Status should read `Server OK`.
2. Type a prompt, set **Duration (frames)**, press **Generate**. The status under the node goes Queued → Running → Done and the node recooks. A 3 s clip takes about 40 s on an RTX 4090; identical requests return from cache instantly.
3. Wire a **Joint Deform**: outputs 0, 1, 2 into inputs 0, 1, 2. Scrub.
4. To steer: connect a curve to input 0 for a root path, or press **Create Pose Rig**, pose it, list the frames in **Pose Keyframes**.
5. For several actions in one clip press **Open Timeline**: right-click the prompt row to add segments, drag their edges to time them, set **Transition**, and add keys on the Full Body or limb tracks (targets come from the posed rig on input 1). **Generate** from the panel or the node. Design notes: [docs/timeline-design.md](docs/timeline-design.md).
6. To put the motion on your own character: **Biped Setup** on both skeletons, **Biped Retarget**, then your deformer. Feed output 3 (T-Pose) through a **Rig Stash Pose** to give Biped Setup a clean rest pose; do not let it synthesise one from the walk.

Parameter by parameter: [houdini/README.md](houdini/README.md).

<!-- WRITING PROMPTS -->
## Writing Prompts

Prompt phrasing has far more effect on the result than any parameter on the node. The numbers below come from measuring generated clips, not from impressions: *airborne* is the highest moment both toes leave the ground, *hip min* is pelvis height against a standing height of ~0.98, *heading* is net rotation over a segment.

### Name the body mechanics, not the intent

The single highest-leverage change. Same model, same 2.0 s duration, nine words rewritten:

| Prompt | Airborne |
|---|---|
| `a person jumps forward and lands on both feet` | 0.217 |
| `a person leaps high into the air with both feet off the ground` | **0.940** |

A 4.3x difference. The same pattern holds elsewhere: `crouches low to the ground` reaches hip 0.722, while `squats down deeply, knees bent, hips near the floor` reaches **0.237**. Describe what the body does; the model does not infer it from the verb.

### A weak prompt collapses inside a timeline; a strong one survives

A segment spends its opening transitioning out of the previous motion, so it has less time left for its own action. A prompt with margin absorbs that; a marginal one does not:

| Prompt | Standalone | In a timeline |
|---|---|---|
| `a person jumps forward and lands on both feet` | 0.217 | 0.122 |
| `a person leaps high into the air with both feet off the ground` | 0.940 | **0.900** |
| `a person stands up and turns around` (no transition segment before it) | -173 deg | **+18 deg** |
| `a person turns around to face the opposite direction` (after its own transition segment) | -193 deg | **-186 deg** |

The strong versions keep about 96% of their standalone result. The weak ones lose most of it. So a timeline does not dilute everything by a fixed amount, it exposes prompts that had no margin to begin with.

This is not caused by whatever precedes the segment: after a neutral standing segment the turn still managed only -73 deg, versus -64 deg after a deep squat.

**So validate a segment standalone first, then assemble.** If it is marginal alone it will disappear in a suite.

### Fix a weak segment by splitting it, not by lengthening it

Duration is segment-dependent, so there is no general rule:

| Fix | Result |
|---|---|
| turn as one 2.25 s segment | -64 deg |
| turn as one 3.50 s segment | -154 deg |
| `stands up from a squat` + `turns around to face the opposite direction` | **-193 deg** |

Splitting beat the same total time spent on one segment. Note the opposite happens for ballistic actions: doubling the jump from 2.0 s to 4.0 s made it *worse* (0.217 to 0.145), because the model spreads the described action across whatever duration it is given.

### The same prompt does not give the same take twice

Kimodo is unseeded, so every generation is a different take and the run-to-run spread can be larger than any prompt change. The same jump prompt, in the same position, in suites whose preceding segments were identical:

| Run | Airborne | Hip peak |
|---|---|---|
| generated alone | 0.940 | - |
| in a suite | 0.900 | 1.468 |
| in a suite, regenerated | **0.192** | **1.030** |

Nothing changed but the take. A good prompt raises the average, it does not guarantee the result.

So for anything you have to show or ship: **generate, measure the segment you care about, and regenerate if it came up weak.** Do not generate once and assume it holds. `force` bypasses the cache to get a fresh take of an identical request, and you can re-roll a single segment rather than the whole clip: see [Regenerating part of a clip](#regenerating-part-of-a-clip).

### Do not prompt for hand or finger detail

Kimodo predicts on the 30-joint `somaskel30`, which strips most finger and hand detail, and converts to the 77-joint skeleton on output. Finger joints in the result are reconstructed, never predicted: `LeftHandIndex2` relative to `LeftHand` measured 0.09882 at frame 1 and 0.09883 at frame 100. Prompts about gestures, grips or finger poses cost generation time and change nothing. When retargeting, set **Blend Fingers** to 0 on Biped Retarget for the same reason.

<!-- PARTIAL REGENERATION -->
## Regenerating Part of a Clip

One weak segment does not mean regenerating the whole clip. Each segment in the __Segments__ multiparm has two buttons, and the timeline panel offers the same pair on right-click:

__Regenerate__ re-rolls that segment alone. The clip keeps its exact length and every frame either side is untouched, so it is a drop-in replacement for one take.

__From Here__ re-rolls that segment and every segment after it, for when the change should carry through the rest of the clip.

### Why not just press Generate again

Because generation is unseeded, so __Generate__ re-rolls *everything*. If eight segments out of nine are good and one is not, a full generation gambles the eight to fix the one. Regenerating a single segment leaves every other frame byte-identical, which is the thing you actually want, and it is the reason to use it even when a full generation is fast.

Speed is a secondary argument, and how secondary depends on [where the text encoder runs](#where-to-run-the-text-encoder). Same nine-segment clip, same machine:

| | Encoder on CPU | Encoder on GPU |
|---|---|---|
| Generate, all nine | 506.9 s | 22.9 s |
| Regenerate one segment | ~60 s | **4.3 s** |

On CPU the time saving is the headline. On GPU both are quick, and preserving the rest of the clip is the whole point.

### Reading the result

The node's __Status__ reports the measured join every time, so you can judge a result rather than assume it:

```
Regenerated segment 5; seam 2.44 / 3.46 cm vs 11.07 cm/sample (0.31x)
```

Two figures for a single segment, one for each join, then the clip's own average movement per sample and the worst join as a ratio of it. Below 1.0x means the join moves less than the motion around it, which is the point at which it stops reading as a cut. Something approaching or above 1.0x is worth looking at.

### How it works, and why a constraint is not enough

Kimodo builds a multi-prompt clip one segment at a time, each joined to the previous one by a transition: the previous tail is prepended as observed motion, moved to the origin, generated against, then moved back and alpha-blended. A fresh `/generate` starts with that history empty, so its first segment takes the "first motion" path and no transition runs.

Handing the previous tail over as an ordinary world-space constraint does not substitute for it. Measured, the constrained frames came back accurate to 0.04 cm but the motion after them continued **184 cm** away, because generation happens in the transition's local frame and a world-space constraint fights it.

So the server takes a `continue_from` tail and seeds that history instead, which needs the `initial_motion` parameter added to `_multiprompt` in `kimodo/model/kimodo_model.py`. That parameter is purely additive: without it the model behaves exactly as before.

Holding the *end* is a separate problem, because whatever follows was generated against the old tail. A `fullbody-global` plus `ee-global` constraint on the new segment's last frames fixes it, and it works here where the same constraint failed above, because with the transition running user constraints are concatenated into the same observed-motion block and translated with it. Unpinned, that join measured 335 cm; pinned, 3.77 cm.

### Requirements and limits

- Needs `initial_motion` in `kimodo/model/kimodo_model.py`, which the [fork](https://github.com/healkeiser/kimodo) used in [Installation](#server) already has. Against stock `nv-tlabs/kimodo` the server returns 422 and the node tells you to update it.
- The timeline must still describe the clip on disk. Edit a segment length and the node refuses until you press __Generate__, because the cut would otherwise land in the wrong place.
- The first segment has no earlier motion to continue from, so both buttons are disabled on it.
- Join quality depends on how dynamic the motion is where it joins: 0.21x joining into a settle, 1.83x joining straight after a jump.

<!-- ENVIRONMENT VARIABLES -->
## Environment Variables

Read by `docker-compose.bridge.yaml`:

| Variable | Default | Purpose |
|----------|---------|---------|
| `HF_HOME` | `$HOME/.cache/huggingface` | Host folder mounted as the Hugging Face cache. Set explicitly on Windows. |
| `HUGGING_FACE_HUB_TOKEN` | — | Token for gated downloads (the Llama text encoder). |
| `KIMODO_MODEL` | Kimodo default | Checkpoint preloaded by the `api` container. |
| `KIMODO_PORT` | `8001` | API port. 8000 is taken by Docker Desktop on Windows. |
| `MOCK_MODE` | `0` | `1` serves `output/dev_reference.npz` (a Kimodo NPZ) without inference, for HDA work without a GPU. Mock clips are cached apart from real ones. |
| `FXMOTION_ROOT` | (required) | The fxhoudinimotion clone, mounted read-only into the `api` container at `/fxmotion`. |
| `FXMOTION_IDLE_UNLOAD_S` | `0` | Seconds without a job before the server frees the model's VRAM; `0` keeps it loaded. |
| `FXMOTION_MOCK_CLIP` | `output/dev_reference.npz` | The Kimodo NPZ served in mock mode. |
| `HF_HUB_OFFLINE` | `1` | Load weights from the local cache only; set `0` for a one-time download. |
| `TEXT_ENCODERS_DIR` | — | Local folder of LLM2Vec adapters, when you cannot pull Meta's repo directly. |
| `KIMODO_ENCODE_EST_S` | `30` | Starting guess, in seconds, for how long a segment spends encoding text. It only shapes the progress bar through the phase Kimodo reports nothing for. The server times the first encode it completes and uses that measurement from then on, including for later jobs, so this matters mainly for the very first generation after a restart. The default is the measured CPU figure. |
| `TEXT_ENCODER_DEVICE` | `cpu` | Where the text encoder runs. `cpu` keeps VRAM free at the cost of ~14 GB of host RAM and the slowest part of every generation; `cuda` is much faster but wants roughly 14 GB of VRAM on top of the motion model. See [Where to run the text encoder](#where-to-run-the-text-encoder). |

Read by the ARDY server (`scripts/run_ardy_server.ps1`):

| Variable | Default | Purpose |
|----------|---------|---------|
| `ARDY_MODEL` | `ARDY-Core-RP-20FPS-Horizon40` | Checkpoint preloaded at start. |
| `TEXT_ENCODER_URL` | `http://127.0.0.1:9550/` | The Kimodo text-encoder container ARDY uses. |
| `OUTPUT_DIR` | `<system temp>/fxmotion/ardy` | Where the server caches clips. |
| `MOCK_MODE` | `0` | `1` serves `FXMOTION_MOCK_CLIP` (an ARDY NPZ) without inference. |
| `FXMOTION_IDLE_UNLOAD_S` | `0` | As for Kimodo. |

The request option `history_s` (default 4) sets how many seconds of motion each ARDY step sees: shorter follows a prompt change sooner, with harder transitions.

### Where to run the text encoder

The text encoder (Llama 3 8B behind LLM2Vec) is the largest thing Kimodo loads and the slowest part of a generation. Where it runs is the main hardware trade-off in this project.

| | `TEXT_ENCODER_DEVICE=cpu` (default) | `TEXT_ENCODER_DEVICE=cuda` |
|---|---|---|
| VRAM | ~4 GB, 6 GB sharing with Houdini | ~17 GB, plus the viewport |
| System RAM | **~14 GB**, held for as long as the container runs | modest |
| Speed | ~30 s of encoding per segment, against ~8 s of denoising | encoding all but disappears |

The default is CPU because it runs on a 6 GB card. If you have VRAM to spare, moving it is the single biggest change you can make. Measured on the same nine-segment, 600-sample clip, same machine, cache bypassed:

| | Encoder on CPU | Encoder on GPU |
|---|---|---|
| Generation | 506.9 s | **22.9 s** |
| Host RAM held | 14.22 GiB | 964 MiB |
| VRAM used | 6.8 GB (viewport only) | 21.7 GB |

Text encoding was not a contributor to generation time, it was generation time. Moving it also hands back about 13 GB of system RAM.

```shell
TEXT_ENCODER_DEVICE=cuda docker compose -f docker-compose.bridge.yaml up -d text-encoder
```

Both containers reserve the GPU, so nothing else needs changing, and no Houdini restart is involved: the encoder is a server-side container the node only talks to over HTTP. Set it in `.env` so it survives a restart.

Check your headroom first. On the 24 GB card above, Houdini was already holding 6.8 GB and the encoder left **2.4 GB free**. That is enough for a viewport but not for a heavy scene or a GPU render alongside it. `nvidia-smi` tells you what you actually have, and a CUDA OOM surfaces as a failed job with the error in the node's __Status__, not as a silent fallback.

Switching costs a container restart: about 100 s warm, and the healthcheck allows up to 10 minutes for a cold CPU load.

**On Windows**, note that the containers run inside WSL2, which by default may claim up to half your RAM and does not hand freed pages back to Windows. With the encoder on CPU that is ~14 GB that looks permanently gone. A `%USERPROFILE%\.wslconfig` fixes it:

```ini
[wsl2]
memory=20GB
autoMemoryReclaim=gradual
```

It takes effect after `wsl --shutdown`, which stops Docker, so do it between sessions.

The Houdini package (`fxhoudinimotion.json`) adds `houdini/` to `HOUDINI_PATH` (otls, python_panels) and `houdini/python` to `PYTHONPATH` (the timeline panel's code).

<!-- DEVELOPMENT -->
## Development

The HDA is generated, not hand-edited. `scripts/build_hda.py` builds the node interface; every callback and cook is a one-line call into `houdini/python/fxmotion/nodes/`, where the logic lives and is tested. `houdini/otls/vb_kimodo_motion_2.0.hda/` is the expanded, VCS-friendly result; the 1.1 asset next to it is frozen.

```shell
# 1. embedded skin mesh + A-pose skeleton (needs the kimodo repo cloned alongside)
hython scripts/build_skin.py

# 2. the HDA, help card included, expanded into houdini/otls/
hython scripts/build_hda.py
```

Timeline panel: `houdini/python/fxmotion/timeline/` (`model.py` is pure Python, run `python tests/test_timeline_model.py`; `bridge.py` talks to the node; `widget.py` is the PySide6 view) and `houdini/python_panels/fxmotion_timeline.pypanel`. The model servers are in `server/`: `fxmotion_server.py` is the shared contract, `kimodo_backend.py` and `kimodo_adapter.py` the Kimodo half. Clips use the `fxmotion.clip/1` format, `houdini/python/fxmotion/clipformat.py`.

In a running Houdini, reload with `hou.hda.reloadFile(...)` and call `matchCurrentDefinition()` on existing nodes. Anything changed in Type Properties by hand is overwritten on the next rebuild, so fold it into the script instead. The node icon is `houdini/config/Icons/kimodo_motion.svg`; icons there are found by bare file name.

<!-- CONTRIBUTING -->
## Contributing

Issues and pull requests are welcome. [CONTRIBUTING.md](CONTRIBUTING.md) covers which half
of the stack a change belongs in, how to test without a GPU (`MOCK_MODE=1`), and the one
rule that costs the most time when missed: the HDA is generated by
`scripts/build_hda.py`, so anything edited by hand in Type Properties is lost on the next
rebuild.

<!-- CREDITS -->
## Credits

- [chordee/kimodo-houdini-bridge](https://github.com/chordee/kimodo-houdini-bridge): the original server, HDA, skinning and transform-convention work this fork builds on.
- [NVIDIA Kimodo](https://github.com/nv-tlabs/kimodo) and the [SOMA](https://research.nvidia.com/labs/sil/projects/kimodo/) skeleton.
- [McGill-NLP LLM2Vec](https://github.com/McGill-NLP/llm2vec) for the text encoder.

<!-- CONTACT -->
## Contact

Project Link: [fxhoudinimotion](https://github.com/healkeiser/fxhoudinimotion)

<p align='center'>
  <!-- GitHub profile -->
  <a href="https://github.com/healkeiser">
    <img src="https://img.shields.io/badge/healkeiser-181717?logo=github&style=social" alt="GitHub"/></a>&nbsp;&nbsp;
  <!-- LinkedIn -->
  <a href="https://www.linkedin.com/in/valentin-beaumont">
    <img src="https://img.shields.io/badge/Valentin%20Beaumont-0A66C2?logo=linkedin&style=social" alt="LinkedIn"/></a>&nbsp;&nbsp;
  <!-- Behance -->
  <a href="https://www.behance.net/el1ven">
    <img src="https://img.shields.io/badge/el1ven-1769FF?logo=behance&style=social" alt="Behance"/></a>&nbsp;&nbsp;
  <!-- X -->
  <a href="https://twitter.com/valentinbeaumon">
    <img src="https://img.shields.io/badge/@valentinbeaumon-1DA1F2?logo=x&style=social" alt="Twitter"/></a>&nbsp;&nbsp;
  <!-- Instagram -->
  <a href="https://www.instagram.com/val.beaumontart">
    <img src="https://img.shields.io/badge/@val.beaumontart-E4405F?logo=instagram&style=social" alt="Instagram"/></a>&nbsp;&nbsp;
  <!-- Gumroad -->
  <a href="https://healkeiser.gumroad.com/subscribe">
    <img src="https://img.shields.io/badge/healkeiser-36a9ae?logo=gumroad&style=social" alt="Gumroad"/></a>&nbsp;&nbsp;
  <!-- Gmail -->
  <a href="mailto:valentin.onze@gmail.com">
    <img src="https://img.shields.io/badge/valentin.onze@gmail.com-D14836?logo=gmail&style=social" alt="Email"/></a>&nbsp;&nbsp;
  <!-- Buy me a coffee -->
  <a href="https://www.buymeacoffee.com/healkeiser">
    <img src="https://img.shields.io/badge/Buy Me A Coffee-FFDD00?&logo=buy-me-a-coffee&logoColor=black" alt="Buy Me A Coffee"/></a>&nbsp;&nbsp;
</p>

## License

Two sets of terms, because this is a fork. See [LICENSE](LICENSE) for the file-by-file split.

- **The timeline panel, the tests, the packaging and the docs written for this fork** are [MIT](LICENSE).
- **Files derived from [chordee/kimodo-houdini-bridge](https://github.com/chordee/kimodo-houdini-bridge)**, which include the model servers in `server/`, the node code ported from the 1.1 asset, the generated HDAs and two of the docs (see [LICENSE](LICENSE) for the full list), carry that project's terms: personal and research use. Upstream ships no license file, so those are the only terms granted, and only its author can widen them.

Taken as a whole this repository is **not** open source under the OSI definition. A grant limited to personal and research use does not meet it.

The model weights are separate again and bind you regardless: the Kimodo weights under the [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license), the text encoder's base weights under the Llama 3 Community License. Read both before any use beyond personal testing.
