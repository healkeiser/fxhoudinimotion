# Phase 1: fxmotion foundation and Kimodo 2.0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the shared fxmotion foundation (clip format, skeleton registry, shared model server, Houdini library) and rebuild Kimodo on it as `vb::kimodo_motion::2.0`, whose animated output matches 1.1 numerically.

**Architecture:** A model server is a `Backend` subclass on the shared FastAPI module `server/fxmotion_server.py`; it returns clips in the `fxmotion.clip/1` NPZ format, converted on the server by a pure adapter (`server/kimodo_adapter.py`). Houdini reads clips through one loader (`fxmotion.clip`), and the HDA's callbacks and cooks are one-line calls into `fxmotion.nodes`, so all logic is version-controlled Python with tests instead of script strings inside the asset.

**Tech Stack:** Python 3.10 (server, inside the kimodo:1.0 image: NGC pytorch 24.10, fastapi 0.135.1, pydantic 2.12.5) and 3.11 (Houdini 22.0), numpy, FastAPI + pydantic v2, requests, Houdini HOM / KineFX, pytest 9, ruff 0.16.7.

**Spec:** `superpowers/specs/2026-09-29-fxhoudinimotion-design.md` (phase 1 of section 7). Phase 0 spikes and phases 2 to 6 get their own plans.

## Global Constraints

- Plans and specs live under `./superpowers/`, never `./docs/`.
- Icons live in `houdini/config/Icons/` and are referenced by bare file name (`kimodo_motion.svg`, `NVIDIA_badge.svg`).
- Source is ASCII only: non-ASCII characters in strings are written as `\u` escapes, comments are ASCII.
- Server code must run on Python 3.10 (`from __future__ import annotations`, no 3.11-only syntax); ruff `target-version = "py310"`, `line-length = 80`.
- Pure modules never import `hou` at module level: `fxmotion/__init__.py`, `fxmotion/clipformat.py`, `fxmotion/skeletons/`, `fxmotion/clip.py`, `fxmotion/paths.py`, `fxmotion/client.py`, `fxmotion/timeline/model.py`, `fxmotion/timeline/regen.py`, everything in `server/`.
- The package variable is `FXMOTION_ROOT`. `KIMODO_BRIDGE_ROOT` is not read anywhere.
- New node type: `vb::kimodo_motion::2.0`. The `vb_kimodo_motion_1.1.hda` directory under `houdini/otls/` is left byte-for-byte untouched.
- Clip format keys, shapes and conventions exactly as spec section 3, plus the clarifications below.
- HDA callbacks and SOP cooks are one-line calls into `fxmotion.nodes`; no logic in HDA script strings.
- Every commit passes `python -m pytest -q`, `uvx ruff@0.16.7 check .` and `uvx ruff@0.16.7 format --check .`. The code blocks in this plan are correct but not necessarily laid out the way ruff wants (import grouping, line wrapping): each commit step first runs `uvx ruff@0.16.7 check --fix . && uvx ruff@0.16.7 format .`, which owns that layout, then the two checks. Commit messages follow the repo style `type(scope): lowercase summary` and end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Spec clarifications decided while planning

These refine the approved spec; the final task records them in the spec file.

1. `effectors` is folded into `keyframes`: a keyframe has `joints: null` (whole body) or a list of joint names (end effectors). Kimodo's end-effector constraint takes the whole pose anyway.
2. `GenerateRequest.options` carries backend-specific knobs. Kimodo uses `transition_frames`, `canon` (reuse a clip's canonical frame, for Regenerate) and `native_constraints` (raw Kimodo constraint dicts in model space, for the Constraints JSON parms and the regen tail pin).
3. A capability `continue` says whether `continue_from` is supported.
4. `root_path` points have an optional `time_s`; all or none. Untimed points are spread over the clip by the server, the way 1.1 spread them.
5. Root path canonicalisation moves from Houdini to the Kimodo adapter, and keyframes are canonicalised with it. In 1.1 pose keys were sent in Houdini space next to a canonicalised root path, so the two disagreed whenever both were used; 2.0 fixes that.
6. The floor invariant (lowest joint within 5 cm of y = 0) applies to generated clips, not to rest poses: the SOMA77 rest pose is hips-centred (feet at y = -1.005).
7. Rotation orthonormality tolerance is 1e-2: Kimodo's float32 global rotations drift 4.5e-3 over a 50-sample clip.
8. Format rule: joints are stored in parent-first order (every parent index is lower than its child's).
9. `vb::kimodo_motion::1.1` still cooks the NPZs it already downloaded, but can no longer Generate: the server contract changed and `kimodo_timeline` is gone.
10. The Kimodo container mounts this repo read-only at `/fxmotion` and starts `uvicorn kimodo_backend:make_app --factory`; server files are no longer copied into the kimodo checkout.
11. The `source_fps` parm is gone: the clip's own `fps` is used.
12. Pose keys with nothing wired to input 1 now raise an error; 1.1 silently ignored them when no timeline was used.

## Review Focus

1. A user points **Clip Path** at a Kimodo-native NPZ saved by 1.1: the node must say it is not an fxmotion clip and name the missing keys, not fail with a KeyError deep in the cook. Pinned in Task 2 (`test_a_kimodo_native_npz_is_rejected_by_name`) and Task 6 (`test_load_names_what_is_wrong_with_a_native_npz`).
2. **Download Dir** does not exist yet (fresh `$HIP`): the download creates it. Pinned in Task 7 (`test_download_creates_the_folder`).
3. Scene at 24 or 25 fps, Start Frame not 1, Retime on and off: the sample shown and the times sent must match 1.1. Pinned in Task 6 (`test_sample_index_matches_v11`) and Task 9 (`test_scene_seconds_matches_v11_samples`).
4. A **non-ASCII prompt** (`\u00e9`, `\u4e00`): the request, the cache key and a second identical request (cache hit) all work. Pinned in Task 4 (`test_non_ascii_prompt_is_cached`).
5. A root path whose first points coincide, or a single point: the canonical heading falls back to +Z instead of dividing by zero. Pinned in Task 3 (`test_canon_ignores_coincident_points`).

---

## File Structure

```
houdini/python/fxmotion/
  __init__.py            package doc; importable without hou
  clipformat.py          fxmotion.clip/1: make, validate, save, load, meta (pure)
  skeletons/__init__.py  Skeleton dataclass, get(name), names() (pure)
  skeletons/soma77.py    SOMA77 data table (moved from scripts/_soma77.py)
  skeletons/data/        soma77_skin.bgeo.sc, soma77_apose.bgeo.sc
  clip.py                clip -> KineFX geometry; math pure, writers take hou geo
  paths.py               path thinning, scene frame -> seconds (pure)
  client.py              HTTP to a model server (requests, no hou)
  qt.py                  Qt binding shim (moved from kimodo_timeline)
  poller.py              JobWatcher on Houdini's event loop (moved, uses client)
  timeline/              the panel (moved from kimodo_timeline): model, bridge, widget, regen
  nodes/common.py        shared generator plumbing: jobs, errors, cooks (hou)
  nodes/kimodo.py        Kimodo Motion node: parms and inputs -> request (hou)
  nodes/timeline_parms.py  Segments multiparm <-> timeline_json (hou)
server/
  fxmotion_server.py     shared FastAPI app: request models, Backend, jobs, cache, idle unload
  kimodo_adapter.py      Kimodo <-> fxmotion conversions (pure numpy)
  kimodo_backend.py      Kimodo inference on the shared server; make_app() factory
scripts/build_hda.py     hython: builds vb::kimodo_motion::2.0 into houdini/otls/
houdini/config/Icons/kimodo_motion.svg
houdini/python_panels/fxmotion_timeline.pypanel
tests/fixtures/kimodo_stop.npz  a real 50-sample Kimodo NPZ
tests/_reference_v11.py  the 1.1 cook math, frozen, for parity tests
```

Removed by the end: `kimodo_server.py`, `scripts/create_hda.py`, `scripts/_add_help.py`, `scripts/_soma77.py`, `scripts/kimodo_icon.svg`, `houdini/python/kimodo_timeline/`, `houdini/python_panels/kimodo_timeline.pypanel`.

---

### Task 1: fxmotion package and the SOMA77 skeleton registry

**Files:**
- Create: `houdini/python/fxmotion/__init__.py`, `houdini/python/fxmotion/skeletons/__init__.py`
- Move: `scripts/_soma77.py` -> `houdini/python/fxmotion/skeletons/soma77.py`
- Create: `houdini/python/fxmotion/skeletons/data/soma77_skin.bgeo.sc`, `.../soma77_apose.bgeo.sc` (decoded from the 1.1 HDA sections)
- Modify: `pyproject.toml`, `.gitattributes`, `scripts/build_skin.py`
- Test: `tests/test_skeletons.py`, `tests/test_imports.py`

**Interfaces:**
- Produces: `fxmotion.skeletons.Skeleton` (fields `name: str`, `joint_names: tuple[str, ...]`, `parents: tuple[int, ...]`, `rest_pos: np.ndarray (J,3) float64`, `rest_rot: np.ndarray (J,3,3) float64, Houdini row-vector`, `skin: Path | None`, `capture_pose: Path | None`; method `index(name) -> int`), `fxmotion.skeletons.get(name: str) -> Skeleton`, `fxmotion.skeletons.names() -> tuple[str, ...]`. `tests/test_imports.py` holds `PURE`, a list later tasks extend.

- [ ] **Step 1: Point pytest at the two source roots**

In `pyproject.toml`, replace the `[tool.pytest.ini_options]` table with:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
# fxmotion lives in houdini/python and the model servers in server/; both are
# plain directories on PYTHONPATH in production (the Houdini package file, the
# container's PYTHONPATH), so the tests see them the same way.
pythonpath = ["houdini/python", "server"]
# test_houdini_live.py imports hou at module level and drives real widgets, so
# it only runs inside a running Houdini session, not hython and not here.
# CONTRIBUTING.md says how to run it.
addopts = "--ignore=tests/test_houdini_live.py"
```

In the same file, change the isort block to:

```toml
[tool.ruff.lint.isort]
known-first-party = ["fxmotion", "fxmotion_server", "kimodo_adapter", "kimodo_backend"]
```

and in `[tool.ruff.format]` replace `exclude = ["scripts/_soma77.py", "*.md"]` with `exclude = ["houdini/python/fxmotion/skeletons/soma77.py", "*.md"]`, and in its comment replace the sentence starting "It is also embedded verbatim as the HDA's PythonModule section" with "The 1.1 HDA embeds its own frozen copy, so this file can change without touching it."

- [ ] **Step 2: Write the failing tests**

`tests/test_skeletons.py`:

```python
"""The skeleton registry: one source of truth for joints, rest poses and rest
geometry, shared by the model servers and the Houdini nodes."""

import numpy as np
import pytest

from fxmotion import skeletons


def test_soma77_is_one_tree_in_parent_first_order():
    s = skeletons.get("soma77")
    assert len(s.joint_names) == len(s.parents) == 77
    assert [i for i, p in enumerate(s.parents) if p < 0] == [0]
    assert all(p < i for i, p in enumerate(s.parents) if p >= 0)
    assert s.index("Hips") == 0


def test_soma77_rest_rotations_are_orthonormal():
    r = skeletons.get("soma77").rest_rot
    assert r.shape == (77, 3, 3)
    err = np.abs(r @ np.transpose(r, (0, 2, 1)) - np.eye(3)).max()
    assert err < 1e-5


def test_soma77_rest_geometry_ships_with_the_package():
    s = skeletons.get("soma77")
    assert s.skin.is_file() and s.skin.stat().st_size > 100_000
    assert s.capture_pose.is_file()


def test_unknown_skeleton_names_the_known_ones():
    with pytest.raises(KeyError, match="soma77"):
        skeletons.get("nope")


def test_get_returns_the_same_object():
    assert skeletons.get("soma77") is skeletons.get("soma77")
```

`tests/test_imports.py`:

```python
"""Pure modules must import without Houdini: the model servers and CI have no
hou. Later tasks append to PURE as they add modules."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PURE = ["fxmotion", "fxmotion.skeletons"]


@pytest.mark.parametrize("module", PURE)
def test_imports_without_hou(module):
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO / "houdini" / "python"), str(REPO / "server")]
    )
    # sys.modules[name] = None makes `import name` raise ImportError
    code = "import sys; sys.modules['hou'] = None; import %s" % module
    subprocess.run([sys.executable, "-c", code], check=True, env=env)
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `python -m pytest tests/test_skeletons.py tests/test_imports.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'fxmotion'`.

- [ ] **Step 4: Move the data table and create the package**

```bash
mkdir -p houdini/python/fxmotion/skeletons/data
git mv scripts/_soma77.py houdini/python/fxmotion/skeletons/soma77.py
```

`houdini/python/fxmotion/__init__.py`:

```python
"""fxmotion: NVIDIA motion models (Kimodo, ARDY, MotionBricks) in Houdini.

Importable without Houdini. The model servers import fxmotion.clipformat and
fxmotion.skeletons, and the offline tests import every module that does not
need hou. Modules that need Houdini live under fxmotion.nodes and
fxmotion.timeline, or import hou inside their functions.
"""
```

`houdini/python/fxmotion/skeletons/__init__.py`:

```python
"""Skeletons the models generate on: joints, rest pose and rest geometry.

Rotations are stored in Houdini's convention (row-vector, bone-aligned), the
same as a clip's world_rot, so a node can write them straight into the
`transform` attribute. A server that needs the column-vector form transposes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent / "data"


@dataclass(frozen=True, eq=False)
class Skeleton:
    name: str
    joint_names: tuple[str, ...]
    parents: tuple[int, ...]
    rest_pos: np.ndarray  # (J, 3), Houdini space
    rest_rot: np.ndarray  # (J, 3, 3), Houdini row-vector
    skin: Path | None = None  # rest geometry with boneCapture (output 0)
    capture_pose: Path | None = None  # the skeleton the skin is bound to (output 1)

    def index(self, name: str) -> int:
        return self.joint_names.index(name)


def _soma77() -> Skeleton:
    from . import soma77 as d

    # TPOSE_ROTS are column-vector; KineFX wants row-vector, hence the transpose
    tp = np.asarray(d.TPOSE_ROTS, dtype=np.float64).reshape(-1, 3, 3)
    return Skeleton(
        name="soma77",
        joint_names=tuple(d.SOMA77_JOINTS),
        parents=tuple(int(p) for p in d.SOMA77_PARENTS),
        rest_pos=np.asarray(d.NEUTRAL_JOINTS, dtype=np.float64),
        rest_rot=np.transpose(tp, (0, 2, 1)),
        skin=DATA / "soma77_skin.bgeo.sc",
        capture_pose=DATA / "soma77_apose.bgeo.sc",
    )


_BUILDERS = {"soma77": _soma77}
_CACHE: dict[str, Skeleton] = {}


def names() -> tuple[str, ...]:
    return tuple(_BUILDERS)


def get(name: str) -> Skeleton:
    if name not in _BUILDERS:
        raise KeyError(
            "unknown skeleton %r; known: %s" % (name, ", ".join(_BUILDERS))
        )
    if name not in _CACHE:
        _CACHE[name] = _BUILDERS[name]()
    return _CACHE[name]
```

Decode the skin and A-pose from the tracked 1.1 sections (they are base64 text there; the decoded bytes were verified identical to the `build_skin.py` output):

```bash
python -c "
import base64
src = 'houdini/otls/vb_kimodo_motion_1.1.hda/vb_8_8Sop_1kimodo__motion_8_81.1/'
dst = 'houdini/python/fxmotion/skeletons/data/'
for a, b in (('skin', 'soma77_skin'), ('apose', 'soma77_apose')):
    open(dst + b + '.bgeo.sc', 'wb').write(base64.b64decode(open(src + a + '.bgeo.sc', 'rb').read()))
"
```

Append to `.gitattributes`:

```
# Geometry and clips are binary; never normalise their bytes
*.bgeo.sc binary
*.npz binary
```

In `scripts/build_skin.py`: replace `from _soma77 import TPOSE_ROTS` with

```python
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "houdini" / "python"))
from fxmotion.skeletons.soma77 import TPOSE_ROTS  # noqa: E402
```

(the `sys` and `Path` imports are already above it), replace `_OUT_DIR = Path(sys.argv[2]) if len(sys.argv) > 2 else _REPO` with `_OUT_DIR = Path(sys.argv[2]) if len(sys.argv) > 2 else _REPO / "houdini" / "python" / "fxmotion" / "skeletons" / "data"`, the two file names `"skin.bgeo.sc"` / `"apose.bgeo.sc"` in `main()` with `"soma77_skin.bgeo.sc"` / `"soma77_apose.bgeo.sc"`, and in the module docstring replace "embedded into the HDA by create_hda.py" with "read by the fxmotion skeleton registry".

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python -m pytest tests/test_skeletons.py tests/test_imports.py -q`
Expected: 7 passed.

- [ ] **Step 6: Lint and commit**

```bash
uvx ruff@0.16.7 check --fix . && uvx ruff@0.16.7 format . && uvx ruff@0.16.7 check . && uvx ruff@0.16.7 format --check .
git add pyproject.toml .gitattributes scripts/build_skin.py houdini/python/fxmotion tests/test_skeletons.py tests/test_imports.py
git commit -m "feat(fxmotion): add the skeleton registry, starting with SOMA77

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: the fxmotion.clip/1 format

**Files:**
- Create: `houdini/python/fxmotion/clipformat.py`
- Create: `tests/fixtures/kimodo_stop.npz` (copy of a real Kimodo output), `tests/cliptools.py`
- Modify: `.gitignore`, `tests/test_imports.py`
- Test: `tests/test_clipformat.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `clipformat.FORMAT = "fxmotion.clip/1"`, `REQUIRED`, `ORTHO_TOL = 1e-2`, `class ClipError(ValueError)`, `make(skeleton, joint_names, parents, fps, world_pos, world_rot, rest_pos, rest_rot, *, contacts=None, segments=None, source=None, native=None) -> dict`, `validate(clip) -> None`, `save(path, clip) -> None`, `load(path) -> dict`, `meta(clip, key) -> dict | list | None`, `frame_count(clip) -> int`. `tests/cliptools.tiny(frames=3, **overrides) -> dict` builds a valid 2-joint clip. Fixture path `tests/fixtures/kimodo_stop.npz` (50 samples, keys `local_rot_mats global_rot_mats posed_joints root_positions smooth_root_pos foot_contacts global_root_heading`).

- [ ] **Step 1: Add the fixture**

The NPZ is a real Kimodo output ("A person stops running and stands still", 50 samples, 320 KB). `*.npz` is git-ignored, so add an exception to `.gitignore` right under the `*.npz` line:

```
!tests/fixtures/*.npz
```

```bash
mkdir -p tests/fixtures
cp ../kimodo/output/fd7a3379235d0475b0ba180e3f382245e24c3066dacc0fd879c743c5a8d86e11.npz tests/fixtures/kimodo_stop.npz
python -c "import numpy as np; d = np.load('tests/fixtures/kimodo_stop.npz'); print(d['posed_joints'].shape)"
```

Expected output: `(50, 77, 3)`.

- [ ] **Step 2: Write the test helper and the failing tests**

`tests/cliptools.py`:

```python
"""Small valid clips for tests that do not care about the skeleton."""

import numpy as np

from fxmotion import clipformat


def tiny(frames=3, **overrides):
    kw = {
        "skeleton": "test",
        "joint_names": ["root", "tip"],
        "parents": [-1, 0],
        "fps": 30.0,
        "world_pos": np.zeros((frames, 2, 3)),
        "world_rot": np.tile(np.eye(3), (frames, 2, 1, 1)),
        "rest_pos": np.zeros((2, 3)),
        "rest_rot": np.tile(np.eye(3), (2, 1, 1)),
    }
    kw.update(overrides)
    return clipformat.make(**kw)
```

`tests/test_clipformat.py`:

```python
"""fxmotion.clip/1: what every server writes and every node reads."""

from pathlib import Path

import numpy as np
import pytest
from cliptools import tiny

from fxmotion import clipformat

FIXTURE = Path(__file__).parent / "fixtures" / "kimodo_stop.npz"


def test_round_trip_keeps_arrays_and_metadata(tmp_path):
    clip = tiny(
        4,
        contacts=np.ones((4, 2)),
        segments=[{"prompt": "walk", "start": 0, "end": 3}],
        source={"backend": "test"},
        native={"extra": np.arange(4)},
    )
    path = tmp_path / "c.npz"
    clipformat.save(path, clip)
    back = clipformat.load(path)
    assert set(back) == set(clip)
    assert np.array_equal(back["world_rot"], clip["world_rot"])
    assert back["contacts"].dtype == np.int8
    assert clipformat.meta(back, "segments")[0]["prompt"] == "walk"
    assert clipformat.meta(back, "source") == {"backend": "test"}
    assert clipformat.meta(back, "missing") is None
    assert clipformat.frame_count(back) == 4
    assert np.array_equal(back["native_extra"], np.arange(4))


def test_a_kimodo_native_npz_is_rejected_by_name():
    with pytest.raises(clipformat.ClipError, match="missing format, skeleton"):
        clipformat.load(FIXTURE)


def test_wrong_format_string():
    clip = tiny()
    clip["format"] = np.array("fxmotion.clip/0")
    with pytest.raises(clipformat.ClipError, match="expected 'fxmotion.clip/1'"):
        clipformat.validate(clip)


def test_exactly_one_root():
    with pytest.raises(clipformat.ClipError, match="exactly one root"):
        tiny(parents=[-1, -1])


def test_parent_first_order():
    with pytest.raises(clipformat.ClipError, match="parent-first"):
        tiny(
            joint_names=["a", "b", "c"],
            parents=[-1, 2, 0],
            world_pos=np.zeros((3, 3, 3)),
            world_rot=np.tile(np.eye(3), (3, 3, 1, 1)),
            rest_pos=np.zeros((3, 3)),
            rest_rot=np.tile(np.eye(3), (3, 1, 1)),
        )


def test_shapes_must_agree():
    with pytest.raises(clipformat.ClipError, match="world_rot has shape"):
        tiny(world_rot=np.tile(np.eye(3), (2, 2, 1, 1)))  # 2 frames, not 3
    with pytest.raises(clipformat.ClipError, match="contacts has shape"):
        tiny(contacts=np.zeros((3, 5)))


def test_nan_is_rejected():
    pos = np.zeros((3, 2, 3))
    pos[1, 1, 2] = np.nan
    with pytest.raises(clipformat.ClipError, match="NaN"):
        tiny(world_pos=pos)


def test_scaled_rotation_is_rejected_but_float32_drift_is_not():
    rot = np.tile(np.eye(3), (3, 2, 1, 1))
    with pytest.raises(clipformat.ClipError, match="not orthonormal"):
        tiny(world_rot=rot * 1.1)
    drift = rot.copy()
    drift[..., 0, 1] = 4.5e-3  # the drift measured on a real Kimodo clip
    tiny(world_rot=drift)


def test_fps_must_be_positive():
    with pytest.raises(clipformat.ClipError, match="fps"):
        tiny(fps=0)
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `python -m pytest tests/test_clipformat.py -q`
Expected: FAIL with `ImportError: cannot import name 'clipformat' from 'fxmotion'`.

- [ ] **Step 4: Write the module**

`houdini/python/fxmotion/clipformat.py`:

```python
"""The fxmotion.clip/1 format: the one shape every model server returns.

Spec: superpowers/specs/2026-09-29-fxhoudinimotion-design.md, section 3.
Pure numpy and no hou, because both sides use it: a server writes clips with
make() and save(), Houdini reads them with load().

Conventions: positions in Houdini space (Y up, metres), rotations in KineFX's
row-vector form and bone-aligned, joints in parent-first order.
"""

from __future__ import annotations

import json

import numpy as np

FORMAT = "fxmotion.clip/1"

REQUIRED = (
    "format",
    "skeleton",
    "joint_names",
    "parents",
    "fps",
    "world_pos",
    "world_rot",
    "rest_pos",
    "rest_rot",
)

# Model rotations are float32 and drift off orthonormal as they compose down
# the chain: 4.5e-3 measured on a 50-sample Kimodo clip. A real mistake, such
# as a matrix carrying a scale, is far larger.
ORTHO_TOL = 1e-2


class ClipError(ValueError):
    """Not a valid fxmotion.clip/1; the message names the offending key."""


def make(
    skeleton,
    joint_names,
    parents,
    fps,
    world_pos,
    world_rot,
    rest_pos,
    rest_rot,
    *,
    contacts=None,
    segments=None,
    source=None,
    native=None,
) -> dict:
    """Assemble and validate a clip. `native` maps names to arrays stored
    as native_<name>: the model's raw data, for its own features."""
    clip = {
        "format": np.array(FORMAT),
        "skeleton": np.array(str(skeleton)),
        "joint_names": np.array([str(n) for n in joint_names]),
        "parents": np.asarray(parents, dtype=np.int32),
        "fps": np.float32(fps),
        "world_pos": np.asarray(world_pos, dtype=np.float32),
        "world_rot": np.asarray(world_rot, dtype=np.float32),
        "rest_pos": np.asarray(rest_pos, dtype=np.float32),
        "rest_rot": np.asarray(rest_rot, dtype=np.float32),
    }
    if contacts is not None:
        clip["contacts"] = np.asarray(contacts, dtype=np.int8)
    if segments is not None:
        clip["segments"] = np.array(json.dumps(segments))
    if source is not None:
        clip["source"] = np.array(json.dumps(source))
    for name, value in (native or {}).items():
        clip["native_" + name] = np.asarray(value)
    validate(clip)
    return clip


def _shape(clip, key, want) -> None:
    if clip[key].shape != want:
        raise ClipError(
            "%s has shape %s, expected %s" % (key, clip[key].shape, want)
        )


def validate(clip) -> None:
    missing = [k for k in REQUIRED if k not in clip]
    if missing:
        raise ClipError(
            "not an %s clip: missing %s" % (FORMAT, ", ".join(missing))
        )
    if str(clip["format"]) != FORMAT:
        raise ClipError(
            "format is %r, expected %r" % (str(clip["format"]), FORMAT)
        )
    names, parents = clip["joint_names"], clip["parents"]
    count = len(names)
    if parents.shape != (count,):
        raise ClipError(
            "parents has shape %s for %d joints" % (parents.shape, count)
        )
    roots = int((parents < 0).sum())
    if roots != 1:
        raise ClipError("parents must have exactly one root, found %d" % roots)
    for i, p in enumerate(parents):
        if p >= i:
            raise ClipError(
                "joint %d (%s) comes before its parent %d; joints must be "
                "in parent-first order" % (i, names[i], p)
            )
    if not float(clip["fps"]) > 0:
        raise ClipError("fps must be positive, got %r" % float(clip["fps"]))
    pos = clip["world_pos"]
    if pos.ndim != 3 or pos.shape[1:] != (count, 3) or pos.shape[0] < 1:
        raise ClipError(
            "world_pos has shape %s, expected (T, %d, 3)" % (pos.shape, count)
        )
    frames = pos.shape[0]
    _shape(clip, "world_rot", (frames, count, 3, 3))
    _shape(clip, "rest_pos", (count, 3))
    _shape(clip, "rest_rot", (count, 3, 3))
    if "contacts" in clip:
        _shape(clip, "contacts", (frames, count))
    for key in ("world_pos", "world_rot", "rest_pos", "rest_rot"):
        if not np.isfinite(clip[key]).all():
            raise ClipError("%s holds NaN or inf" % key)
    for key in ("world_rot", "rest_rot"):
        r = clip[key].astype(np.float64)
        err = float(np.abs(r @ np.swapaxes(r, -1, -2) - np.eye(3)).max())
        if err > ORTHO_TOL:
            raise ClipError("%s is not orthonormal (error %.3g)" % (key, err))
    for key in ("segments", "source"):
        if key in clip:
            try:
                json.loads(str(clip[key]))
            except ValueError as e:
                raise ClipError("%s is not JSON: %s" % (key, e)) from e


def meta(clip, key):
    """The JSON stored under `key` (segments, source), or None."""
    return json.loads(str(clip[key])) if key in clip else None


def frame_count(clip) -> int:
    return int(clip["world_pos"].shape[0])


def save(path, clip) -> None:
    """Validate and write. Give `path` an .npz suffix: numpy appends one
    otherwise."""
    validate(clip)
    np.savez_compressed(path, **clip)


def load(path) -> dict:
    """Every array of a clip file, validated."""
    with np.load(path, allow_pickle=False) as z:
        clip = {k: z[k] for k in z.files}
    validate(clip)
    return clip
```

Append `"fxmotion.clipformat"` to `PURE` in `tests/test_imports.py`.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python -m pytest tests/test_clipformat.py tests/test_imports.py -q`
Expected: all pass.

- [ ] **Step 6: Lint and commit**

```bash
uvx ruff@0.16.7 check --fix . && uvx ruff@0.16.7 format . && uvx ruff@0.16.7 check . && uvx ruff@0.16.7 format --check .
git add .gitignore houdini/python/fxmotion/clipformat.py tests/cliptools.py tests/test_clipformat.py tests/test_imports.py tests/fixtures/kimodo_stop.npz
git commit -m "feat(fxmotion): define the fxmotion.clip/1 format every server returns

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: the Kimodo adapter (pure conversions)

**Files:**
- Create: `server/kimodo_adapter.py`, `tests/_reference_v11.py`
- Modify: `tests/test_imports.py`
- Test: `tests/test_kimodo_adapter.py`

**Interfaces:**
- Consumes: `fxmotion.clipformat.make`, `fxmotion.skeletons.get("soma77")`.
- Produces (module `kimodo_adapter`): `FPS = 30.0`; `NATIVE_KEYS`; `class AdapterError(ValueError)`; `yaw(ang) -> np.ndarray (3,3)`; `@dataclass(frozen=True) Canon(ox=0.0, oz=0.0, ang=0.0)` with `Canon.from_xz(xz) -> Canon`, `pos_to_model(p)`, `pos_to_world(p)`, `rot_to_model(r)`, `rot_to_world(r)`, `as_list() -> [ox, oz, ang]`; `sample(time_s) -> int`; `to_clip(npz: dict, canon: Canon, *, segments=None, source=None) -> dict`; `keyframe_constraints(keyframes: list[dict], canon: Canon) -> list[dict]`; `@dataclass KimodoInputs(texts, durations_s, num_frames, constraints, canon, transition_frames, continue_from)`; `kimodo_inputs(req: dict) -> KimodoInputs`, where `req` is a `GenerateRequest.model_dump()` (Task 4). `tests/_reference_v11.cook(npz, frame, ox, oz, ang) -> (pos (J,3), transform (J,9), local (J,16))`.

- [ ] **Step 1: Freeze the 1.1 cook math as a reference**

`tests/_reference_v11.py` (the numbers are what `scripts/create_hda.py`'s `_COOK_SCRIPT` produced; keep it frozen, it is the parity target):

```python
"""The vb::kimodo_motion::1.1 animated cook, frozen as a reference.

Copied from _COOK_SCRIPT in scripts/create_hda.py at commit 047779c (the
last 1.1 build). 2.0 must reproduce these numbers; do not edit this file to
make a test pass.
"""

import numpy as np

from fxmotion.skeletons.soma77 import SOMA77_PARENTS, TPOSE_ROTS


def cook(npz, frame, ox=0.0, oz=0.0, ang=0.0):
    pos = npz["posed_joints"][frame].copy().astype(np.float64)
    grot = npz["global_rot_mats"][frame].astype(np.float64)
    if ang or ox or oz:
        c, s_ = np.cos(ang), np.sin(ang)
        R = np.array([[c, 0.0, s_], [0.0, 1.0, 0.0], [-s_, 0.0, c]])
        pos = pos @ R.T + np.array([ox, 0.0, oz])
        grot = np.einsum("ij,njk->nik", R, grot)
    tp = np.asarray(TPOSE_ROTS, dtype=float).reshape(-1, 3, 3)
    world_rot = [grot[i] @ tp[i] for i in range(len(SOMA77_PARENTS))]
    world_m = []
    for i in range(len(SOMA77_PARENTS)):
        m = np.identity(4)
        m[:3, :3] = world_rot[i].T
        m[3, :3] = pos[i]
        world_m.append(m)
    transforms, locals_ = [], []
    for i, parent in enumerate(SOMA77_PARENTS):
        local_m = (
            world_m[i]
            if parent < 0
            else world_m[i] @ np.linalg.inv(world_m[parent])
        )
        transforms.append(world_rot[i].T.flatten())
        locals_.append(local_m.flatten())
    return pos, np.array(transforms), np.array(locals_)


def canon_xz(xz):
    """1.1's Generate-callback canonicalisation of a root path."""
    import math

    ox, oz = xz[0]
    dx, dz = next(
        (
            (x - ox, z - oz)
            for x, z in xz[1:]
            if abs(x - ox) + abs(z - oz) > 1e-4
        ),
        (0.0, 1.0),
    )
    ang = math.atan2(dx, dz)
    ca, sa = math.cos(-ang), math.sin(-ang)
    pts = [
        [(x - ox) * ca + (z - oz) * sa, -(x - ox) * sa + (z - oz) * ca]
        for x, z in xz
    ]
    return pts, (ox, oz, ang)
```

- [ ] **Step 2: Write the failing tests**

`tests/test_kimodo_adapter.py`:

```python
"""Kimodo <-> fxmotion conversions, against a real Kimodo NPZ and the frozen
1.1 cook."""

from pathlib import Path

import _reference_v11 as v11
import kimodo_adapter as ka
import numpy as np
import pytest

from fxmotion import clipformat

FIXTURE = Path(__file__).parent / "fixtures" / "kimodo_stop.npz"
CANONS = (ka.Canon(), ka.Canon(1.5, -2.0, 0.7))


def _npz():
    with np.load(FIXTURE) as z:
        return {k: z[k] for k in z.files}


def _req(**over):
    req = {
        "segments": [{"prompt": "a person walks", "duration_s": 2.0}],
        "root_path": None,
        "keyframes": None,
        "continue_from": None,
        "seed": None,
        "model": "",
        "force": False,
        "options": {},
    }
    req.update(over)
    return req


@pytest.mark.parametrize("canon", CANONS)
def test_to_clip_matches_the_v11_cook(canon):
    npz = _npz()
    clip = ka.to_clip(npz, canon)
    for frame in (0, 10, 25, 49):
        pos, xform, _ = v11.cook(npz, frame, canon.ox, canon.oz, canon.ang)
        assert np.abs(clip["world_pos"][frame] - pos).max() < 1e-5
        got = clip["world_rot"][frame].reshape(-1, 9)
        assert np.abs(got - xform).max() < 1e-5


def test_to_clip_is_valid_and_keeps_contacts_native_and_canon():
    npz = _npz()
    clip = ka.to_clip(npz, ka.Canon(1.0, 2.0, 0.5), source={"model": "m"})
    clipformat.validate(clip)
    assert str(clip["skeleton"]) == "soma77" and float(clip["fps"]) == 30.0
    names = [str(n) for n in clip["joint_names"]]
    left = names.index("LeftFoot")
    assert np.array_equal(clip["contacts"][:, left], npz["foot_contacts"][:, 0])
    for key in ka.NATIVE_KEYS:
        assert np.array_equal(clip["native_" + key], npz[key])
    src = clipformat.meta(clip, "source")
    assert src == {"model": "m", "backend": "kimodo", "canon": [1.0, 2.0, 0.5]}


def test_generated_clip_reaches_the_floor():
    low = ka.to_clip(_npz(), ka.Canon())["world_pos"][..., 1].min()
    assert -0.05 < low < 0.05


def test_canon_round_trips():
    c = ka.Canon(1.5, -2.0, 0.7)
    p = np.random.default_rng(0).normal(size=(5, 3))
    assert np.allclose(c.pos_to_world(c.pos_to_model(p)), p)
    r = _npz()["global_rot_mats"][3]
    assert np.allclose(c.rot_to_world(c.rot_to_model(r)), r, atol=1e-6)


def test_canon_puts_the_path_start_on_plus_z_like_v11():
    xz = [[2.0, 1.0], [3.0, 3.0], [5.0, 3.5]]
    want, (ox, oz, ang) = v11.canon_xz(xz)
    c = ka.Canon.from_xz(xz)
    assert np.allclose(c.as_list(), [ox, oz, ang])
    got = [c.pos_to_model([x, 0.0, z])[[0, 2]] for x, z in xz]
    assert np.allclose(got, want)
    assert abs(got[1][0]) < 1e-9 and got[1][1] > 0


def test_canon_ignores_coincident_points():
    assert ka.Canon.from_xz([[1.0, 1.0], [1.0, 1.0]]).as_list() == [1.0, 1.0, 0.0]
    assert ka.Canon.from_xz([[4.0, -2.0]]).as_list() == [4.0, -2.0, 0.0]


def test_timed_path_samples_at_30_fps():
    path = [
        {"pos": [0.0, 0.0, 0.0], "time_s": 0.0},
        {"pos": [0.0, 0.0, 2.0], "time_s": 1.0},
    ]
    inp = ka.kimodo_inputs(_req(root_path=path))
    root2d = inp.constraints[-1]
    assert root2d["type"] == "root2d"
    assert root2d["frame_indices"] == [0, 30]
    assert np.allclose(root2d["smooth_root_2d"], [[0.0, 0.0], [0.0, 2.0]])


def test_untimed_path_spreads_over_the_clip_like_v11():
    path = [{"pos": [float(i), 0.0, 0.0]} for i in range(4)]
    inp = ka.kimodo_inputs(_req(root_path=path))  # 2.0 s -> 60 samples
    assert inp.constraints[-1]["frame_indices"] == [0, 20, 39, 59]
    assert np.allclose(inp.canon.as_list(), [0.0, 0.0, np.pi / 2])


def test_mixed_timing_is_refused():
    path = [{"pos": [0.0, 0.0, 0.0], "time_s": 0.0}, {"pos": [1.0, 0.0, 0.0]}]
    with pytest.raises(ka.AdapterError, match="all have time_s or none"):
        ka.kimodo_inputs(_req(root_path=path))


def test_a_keyframe_round_trips_to_kimodo_model_space():
    npz = _npz()
    canon = ka.Canon(1.5, -2.0, 0.7)
    clip = ka.to_clip(npz, canon)
    kf = {
        "time_s": 10 / 30.0,
        "world_pos": clip["world_pos"][10].tolist(),
        "world_rot": clip["world_rot"][10].tolist(),
        "joints": None,
    }
    (c,) = ka.keyframe_constraints([kf], canon)
    assert c["type"] == "fullbody-global" and c["frame_indices"] == [10]
    assert np.abs(np.array(c["global_joints_positions"][0]) - npz["posed_joints"][10]).max() < 1e-4
    assert np.abs(np.array(c["global_joints_rots"][0]) - npz["global_rot_mats"][10]).max() < 1e-4
    hips = npz["posed_joints"][10][0]
    assert np.allclose(c["smooth_root_2d"][0], [hips[0], hips[2]], atol=1e-4)


def test_end_effector_keys_group_by_joints():
    clip = ka.to_clip(_npz(), ka.Canon())
    def kf(t, joints):
        return {"time_s": t, "world_pos": clip["world_pos"][0].tolist(), "world_rot": clip["world_rot"][0].tolist(), "joints": joints}
    out = ka.keyframe_constraints([kf(1.0, ["LeftHand"]), kf(0.5, ["LeftHand"]), kf(0.2, None)], ka.Canon())
    assert [c["type"] for c in out] == ["ee-global", "fullbody-global"]
    assert out[0]["joint_names"] == ["LeftHand"] and out[0]["frame_indices"] == [15, 30]


def test_keyframe_needs_every_joint():
    kf = {"time_s": 0.0, "world_pos": [[0.0, 0.0, 0.0]], "world_rot": [np.eye(3).tolist()], "joints": None}
    with pytest.raises(ka.AdapterError, match="every SOMA77 joint"):
        ka.keyframe_constraints([kf], ka.Canon())


def test_segments_options_and_constraint_order():
    native = [{"type": "root2d", "frame_indices": [0], "smooth_root_2d": [[0, 0]]}]
    inp = ka.kimodo_inputs(
        _req(
            segments=[
                {"prompt": " walk ", "duration_s": 1.0},
                {"prompt": "run", "duration_s": 0.5},
            ],
            root_path=[{"pos": [0.0, 0.0, 0.0]}, {"pos": [0.0, 0.0, 1.0]}],
            options={"transition_frames": 7, "native_constraints": native},
        )
    )
    assert inp.texts == ["walk", "run"] and inp.num_frames == [30, 15]
    assert inp.transition_frames == 7
    assert [c["type"] for c in inp.constraints] == ["root2d", "root2d"]
    assert inp.constraints[0] is native[0]


def test_empty_prompt_is_refused():
    with pytest.raises(ka.AdapterError, match="segment 1 has no prompt"):
        ka.kimodo_inputs(_req(segments=[{"prompt": "  ", "duration_s": 1.0}]))


def test_canon_option_wins_over_the_path():
    inp = ka.kimodo_inputs(_req(options={"canon": [1.0, 2.0, 0.3]}, root_path=[{"pos": [9.0, 0.0, 9.0]}]))
    assert inp.canon.as_list() == [1.0, 2.0, 0.3]


def test_continue_from_maps_the_native_keys():
    cf = {"native_local_rot_mats": [[1]], "native_root_positions": [[2]]}
    inp = ka.kimodo_inputs(_req(continue_from=cf))
    assert inp.continue_from == {"local_rot_mats": [[1]], "root_positions": [[2]]}
    with pytest.raises(ka.AdapterError, match="native_root_positions"):
        ka.kimodo_inputs(_req(continue_from={"native_local_rot_mats": [[1]]}))
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `python -m pytest tests/test_kimodo_adapter.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'kimodo_adapter'`.

- [ ] **Step 4: Write the adapter**

`server/kimodo_adapter.py`:

```python
"""Kimodo <-> fxmotion: the pure half of the Kimodo server.

No torch and no kimodo import, so the conversion is tested offline against a
real Kimodo NPZ. The inference half is kimodo_backend.py.

Two spaces meet here:
  model space  where Kimodo generates: root at XZ (0, 0) facing +Z on the
               first sample; rotations column-vector, world-axis-aligned at
               rest.
  Houdini      where the request and the clip live: the root path as drawn;
               rotations row-vector and bone-aligned, ready for KineFX.
Canon moves positions and rotations between the two frames; to_clip and
keyframe_constraints convert the rotation convention.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from fxmotion import clipformat, skeletons

FPS = 30.0

# Every per-sample array in a Kimodo NPZ, kept as native_* for Regenerate.
NATIVE_KEYS = (
    "local_rot_mats",
    "global_rot_mats",
    "posed_joints",
    "root_positions",
    "smooth_root_pos",
    "foot_contacts",
    "global_root_heading",
)

# Joints behind each foot_contacts channel. A SOMA77 NPZ carries 6 channels
# (ToeEnd duplicates ToeBase); NPZs from the 4-channel internal representation
# carry only the detected pair.
FOOT_CHANNELS = {
    6: (
        "LeftFoot",
        "LeftToeBase",
        "LeftToeEnd",
        "RightFoot",
        "RightToeBase",
        "RightToeEnd",
    ),
    4: ("LeftFoot", "LeftToeBase", "RightFoot", "RightToeBase"),
}


class AdapterError(ValueError):
    """A request Kimodo cannot run; the server answers 422 with it."""


def yaw(ang: float) -> np.ndarray:
    """Rotation about +Y by `ang`, column-vector (the 1.1 cook's matrix)."""
    c, s = math.cos(ang), math.sin(ang)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


@dataclass(frozen=True)
class Canon:
    """Where model space sits in Houdini: origin (ox, 0, oz), heading ang."""

    ox: float = 0.0
    oz: float = 0.0
    ang: float = 0.0

    @classmethod
    def from_xz(cls, xz) -> Canon:
        """Origin on the first point, heading toward the first point that
        is not on top of it (+Z if there is none)."""
        ox, oz = (float(v) for v in xz[0])
        dx, dz = next(
            (
                (x - ox, z - oz)
                for x, z in xz[1:]
                if abs(x - ox) + abs(z - oz) > 1e-4
            ),
            (0.0, 1.0),
        )
        return cls(ox, oz, math.atan2(dx, dz))

    def _origin(self) -> np.ndarray:
        return np.array([self.ox, 0.0, self.oz])

    def pos_to_model(self, p) -> np.ndarray:
        return (np.asarray(p, dtype=np.float64) - self._origin()) @ yaw(self.ang)

    def pos_to_world(self, p) -> np.ndarray:
        return np.asarray(p, dtype=np.float64) @ yaw(self.ang).T + self._origin()

    def rot_to_model(self, r) -> np.ndarray:
        return yaw(self.ang).T @ np.asarray(r, dtype=np.float64)

    def rot_to_world(self, r) -> np.ndarray:
        return yaw(self.ang) @ np.asarray(r, dtype=np.float64)

    def as_list(self) -> list:
        return [self.ox, self.oz, self.ang]


def sample(time_s) -> int:
    """Clip sample for a time in seconds."""
    return int(round(float(time_s) * FPS))


def _tpose() -> np.ndarray:
    """Column-vector T-pose offsets per joint (1.1's TPOSE_ROTS)."""
    rest = skeletons.get("soma77").rest_rot
    return np.swapaxes(rest.astype(np.float64), -1, -2)


def to_clip(npz, canon: Canon, *, segments=None, source=None) -> dict:
    """A Kimodo output (model space) as an fxmotion clip (Houdini space)."""
    skel = skeletons.get("soma77")
    pos = canon.pos_to_world(npz["posed_joints"])
    grot = canon.rot_to_world(npz["global_rot_mats"])
    # right-multiplying the T-pose offsets makes each frame bone-aligned,
    # matching the T-Pose output; the transpose is KineFX's row-vector form
    world_rot = np.swapaxes(grot @ _tpose(), -1, -2)
    contacts = None
    fc = npz.get("foot_contacts")
    if fc is not None and fc.shape[-1] in FOOT_CHANNELS:
        contacts = np.zeros(pos.shape[:2], dtype=np.int8)
        for ch, name in enumerate(FOOT_CHANNELS[fc.shape[-1]]):
            contacts[:, skel.index(name)] = fc[:, ch]
    return clipformat.make(
        "soma77",
        skel.joint_names,
        skel.parents,
        FPS,
        pos,
        world_rot,
        skel.rest_pos,
        skel.rest_rot,
        contacts=contacts,
        segments=segments,
        source=dict(source or {}, backend="kimodo", canon=canon.as_list()),
        native={k: npz[k] for k in NATIVE_KEYS if k in npz},
    )


def keyframe_constraints(keyframes, canon: Canon) -> list:
    """Request keyframes (Houdini space, KineFX rotations) as Kimodo
    fullbody-global / ee-global dicts in model space. Keys sharing the same
    `joints` become one constraint, like 1.1's one per track."""
    skel = skeletons.get("soma77")
    count = len(skel.joint_names)
    tp_t = np.swapaxes(_tpose(), -1, -2)
    hips = skel.index("Hips")
    groups: dict = {}
    for kf in keyframes:
        pos = np.asarray(kf["world_pos"], dtype=np.float64)
        rot = np.asarray(kf["world_rot"], dtype=np.float64)
        if pos.shape != (count, 3) or rot.shape != (count, 3, 3):
            raise AdapterError(
                "kimodo: a keyframe needs world_pos (%d, 3) and world_rot "
                "(%d, 3, 3) for every SOMA77 joint, got %s and %s"
                % (count, count, pos.shape, rot.shape)
            )
        joints = tuple(kf.get("joints") or ())
        unknown = [j for j in joints if j not in skel.joint_names]
        if unknown:
            raise AdapterError(
                "kimodo: unknown joints %s" % ", ".join(unknown)
            )
        # KineFX transform R = (grot @ tp).T, so grot = R.T @ tp.T
        grot = np.swapaxes(rot, -1, -2) @ tp_t
        groups.setdefault(joints, []).append(
            (sample(kf["time_s"]), canon.pos_to_model(pos), canon.rot_to_model(grot))
        )
    out = []
    for joints, keys in groups.items():
        keys.sort(key=lambda k: k[0])
        c = {
            "frame_indices": [f for f, _, _ in keys],
            "global_joints_positions": [p.tolist() for _, p, _ in keys],
            "global_joints_rots": [r.tolist() for _, _, r in keys],
            "smooth_root_2d": [[p[hips, 0], p[hips, 2]] for _, p, _ in keys],
        }
        if joints:
            c.update(type="ee-global", joint_names=list(joints))
        else:
            c["type"] = "fullbody-global"
        out.append(c)
    return out


def _path_items(points, duration_s):
    """[(sample, [x, z])], deduplicated by sample (first wins), sorted."""
    timed = [p.get("time_s") is not None for p in points]
    if any(timed) and not all(timed):
        raise AdapterError(
            "kimodo: root_path points must all have time_s or none"
        )
    xz = [[float(p["pos"][0]), float(p["pos"][2])] for p in points]
    if all(timed):
        frames = [sample(p["time_s"]) for p in points]
    elif len(xz) > 1:
        # spread evenly over the clip, as 1.1 did
        last = max(2, int(duration_s * FPS)) - 1
        frames = [int(round(i * last / (len(xz) - 1))) for i in range(len(xz))]
    else:
        frames = [0]
    by_frame: dict = {}
    for f, c in zip(frames, xz):
        by_frame.setdefault(f, c)
    return sorted(by_frame.items())


@dataclass
class KimodoInputs:
    texts: list
    durations_s: list
    num_frames: list
    constraints: list
    canon: Canon
    transition_frames: int
    continue_from: dict | None


def kimodo_inputs(req: dict) -> KimodoInputs:
    """A GenerateRequest (as a dict) turned into what Kimodo's model call
    takes. Raises AdapterError for anything Kimodo cannot run."""
    texts, durs = [], []
    for i, seg in enumerate(req.get("segments") or [], start=1):
        text = (seg.get("prompt") or "").strip()
        dur = float(seg.get("duration_s") or 0.0)
        if not text:
            raise AdapterError("kimodo: segment %d has no prompt" % i)
        if dur <= 0:
            raise AdapterError("kimodo: segment %d needs duration_s > 0" % i)
        texts.append(text)
        durs.append(dur)
    if not texts:
        raise AdapterError("kimodo: at least one segment is required")
    opts = req.get("options") or {}
    root_path = req.get("root_path") or []
    items = _path_items(root_path, sum(durs)) if root_path else []
    if opts.get("canon") is not None:
        canon = Canon(*(float(v) for v in opts["canon"]))
    elif items:
        canon = Canon.from_xz([c for _, c in items])
    else:
        canon = Canon()
    native = opts.get("native_constraints") or []
    if not isinstance(native, list):
        raise AdapterError("kimodo: native_constraints must be a list")
    constraints = list(native)
    if items:
        constraints.append(
            {
                "type": "root2d",
                "frame_indices": [f for f, _ in items],
                "smooth_root_2d": [
                    canon.pos_to_model([x, 0.0, z])[[0, 2]].tolist()
                    for _, (x, z) in items
                ],
            }
        )
    constraints += keyframe_constraints(req.get("keyframes") or [], canon)
    cont = None
    cf = req.get("continue_from")
    if cf:
        try:
            cont = {
                "local_rot_mats": cf["native_local_rot_mats"],
                "root_positions": cf["native_root_positions"],
            }
        except KeyError as e:
            raise AdapterError(
                "kimodo: continue_from needs native_local_rot_mats and "
                "native_root_positions (%s missing)" % e
            ) from e
    return KimodoInputs(
        texts=texts,
        durations_s=durs,
        num_frames=[max(1, int(d * FPS)) for d in durs],
        constraints=constraints,
        canon=canon,
        transition_frames=max(1, int(opts.get("transition_frames", 5))),
        continue_from=cont,
    )
```

Append `"kimodo_adapter"` to `PURE` in `tests/test_imports.py`.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python -m pytest tests/test_kimodo_adapter.py tests/test_imports.py -q`
Expected: all pass. If `test_untimed_path_spreads_over_the_clip_like_v11` fails on the frame list, recompute the expectation by hand from 1.1's formula `round(i * (T - 1) / (n - 1))` with `T = int(2.0 * 30) = 60`: `[0, 20, 39, 59]` (`round(39.33) = 39`); fix the code, not the expectation.

- [ ] **Step 6: Lint and commit**

```bash
uvx ruff@0.16.7 check --fix . && uvx ruff@0.16.7 format . && uvx ruff@0.16.7 check . && uvx ruff@0.16.7 format --check .
git add server/kimodo_adapter.py tests/_reference_v11.py tests/test_kimodo_adapter.py tests/test_imports.py
git commit -m "feat(server): convert Kimodo output to fxmotion clips, and requests back

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: the shared model server

**Files:**
- Create: `server/fxmotion_server.py`
- Modify: `.github/workflows/tests.yml`, `tests/test_imports.py`
- Test: `tests/test_server.py`

**Interfaces:**
- Consumes: `fxmotion.clipformat.save`.
- Produces (module `fxmotion_server`): pydantic models `Segment(duration_s: float, prompt: str | None, style: str | None)`, `PathPoint(pos: list[float], time_s: float | None)`, `Keyframe(time_s: float, world_pos: list[list[float]], world_rot: list[list[list[float]]], joints: list[str] | None)`, `GenerateRequest(segments, root_path=None, keyframes=None, continue_from=None, seed=None, model="", force=False, options={})`, `JobStatus`; `class Progress` with `set(fraction, phase=None, est_s=None, span=None)` and property `cancelled`; `class Backend` with class attributes `name, skeleton, fps, models, default_model, capabilities` and methods `validate(req)`, `load(model)`, `unload()`, `generate(req, progress) -> dict`, `mock(req) -> dict`, `describe(req) -> str`; `unsupported(req, backend) -> list[str]`; `create_app(backend, *, output_dir, mock=False, idle_unload_s=0.0, preload=False) -> FastAPI` whose `app.state.reap_idle(now=None) -> bool`. Endpoints: `GET /health`, `POST /generate` (202, 422), `GET /jobs/{id}`, `GET /jobs/{id}/download` (404, 409), `POST /jobs/{id}/cancel`.

- [ ] **Step 1: Write the failing tests**

`tests/test_server.py`:

```python
"""The shared server contract, driven with a fake backend: no model, no GPU."""

import threading
import time

import fxmotion_server as fs
import numpy as np
import pytest
from cliptools import tiny
from fastapi.testclient import TestClient

from fxmotion import clipformat

REQ = {"segments": [{"prompt": "a person walks", "duration_s": 2.0}]}


class FakeBackend(fs.Backend):
    name = "fake"
    skeleton = "test"
    fps = 30.0
    models = ("m1", "m2")
    default_model = "m1"
    capabilities = {
        "text": True,
        "styles": ["walk"],
        "root_path": True,
        "keyframes": False,
        "effectors": False,
        "continue": False,
        "live": False,
    }

    def __init__(self):
        self.calls, self.loaded, self.unloaded = [], [], 0
        self.gate, self.fail = None, None

    def validate(self, req):
        if any(s.prompt == "" for s in req.segments):
            raise ValueError("fake: empty prompt")

    def load(self, model):
        self.loaded.append(model)

    def unload(self):
        self.unloaded += 1

    def generate(self, req, progress):
        if self.gate is not None:
            self.gate.wait(5)
        if self.fail:
            raise RuntimeError(self.fail)
        progress.set(0.5, "working")
        self.calls.append(req)
        return tiny(4)

    def mock(self, req):
        return tiny(2)


def _serve(tmp_path, **kw):
    backend = FakeBackend()
    app = fs.create_app(backend, output_dir=tmp_path, **kw)
    return backend, app


@pytest.fixture
def server(tmp_path):
    backend, app = _serve(tmp_path)
    with TestClient(app) as client:
        yield client, backend, app


def wait(client, job_id, timeout=5.0):
    end = time.monotonic() + timeout
    while True:
        st = client.get("/jobs/%s" % job_id).json()
        if st["status"] in ("done", "failed", "cancelled"):
            return st
        assert time.monotonic() < end, "job still %s" % st["status"]
        time.sleep(0.02)


def submit(client, req=REQ):
    r = client.post("/generate", json=req)
    assert r.status_code == 202, r.text
    return r.json()["job_id"]


def test_health_reports_the_backend(server):
    client, _, _ = server
    h = client.get("/health").json()
    assert h["backend"] == "fake" and h["fps"] == 30.0
    assert h["models"] == ["m1", "m2"] and h["capabilities"]["styles"] == ["walk"]
    assert h["mock_mode"] is False and h["loaded_model"] is None


def test_generate_then_download_a_valid_clip(server, tmp_path):
    client, backend, _ = server
    st = wait(client, submit(client))
    assert st["status"] == "done" and st["frames"] == 4 and st["joints"] == 2
    assert st["cached"] is False and backend.loaded == ["m1"]
    blob = client.get("/jobs/%s/download" % st["job_id"]).content
    path = tmp_path / "got.npz"
    path.write_bytes(blob)
    assert clipformat.frame_count(clipformat.load(path)) == 4


def test_unsupported_controls_are_refused_by_name(server):
    client, _, _ = server
    req = dict(REQ, keyframes=[{"time_s": 0.0, "world_pos": [[0, 0, 0]], "world_rot": [np.eye(3).tolist()]}])
    r = client.post("/generate", json=req)
    assert r.status_code == 422
    assert r.json()["detail"] == "fake: not supported: keyframes"
    r = client.post("/generate", json={"segments": [{"style": "run", "duration_s": 1.0}]})
    assert r.status_code == 422
    assert "style 'run' (known: walk)" in r.json()["detail"]


def test_backend_validation_is_a_422(server):
    client, _, _ = server
    r = client.post("/generate", json={"segments": [{"prompt": "", "duration_s": 1.0}]})
    assert r.status_code == 422 and r.json()["detail"] == "fake: empty prompt"


def test_identical_requests_hit_the_cache_unless_forced(server):
    client, backend, _ = server
    wait(client, submit(client))
    st = wait(client, submit(client))
    assert st["cached"] is True and len(backend.calls) == 1
    wait(client, submit(client, dict(REQ, force=True)))
    assert len(backend.calls) == 2


def test_non_ascii_prompt_is_cached(server):
    client, backend, _ = server
    req = {"segments": [{"prompt": "une personne marche \u00e9 \u4e00", "duration_s": 1.0}]}
    assert wait(client, submit(client, req))["status"] == "done"
    assert wait(client, submit(client, req))["cached"] is True
    assert backend.calls[0].segments[0].prompt.endswith("\u4e00")


def test_a_failure_is_reported(server):
    client, backend, _ = server
    backend.fail = "boom"
    st = wait(client, submit(client))
    assert st["status"] == "failed" and "boom" in st["error"]


def test_cancel_a_queued_job(server):
    client, backend, _ = server
    backend.gate = threading.Event()
    first = submit(client)
    second = submit(client, dict(REQ, force=True))
    assert client.post("/jobs/%s/cancel" % second).json()["status"] == "cancelled"
    backend.gate.set()
    assert wait(client, first)["status"] == "done"
    assert wait(client, second)["status"] == "cancelled"
    assert len(backend.calls) == 1


def test_download_errors(server):
    client, backend, _ = server
    assert client.get("/jobs/nope/download").status_code == 404
    backend.gate = threading.Event()
    job = submit(client)
    assert client.get("/jobs/%s/download" % job).status_code == 409
    backend.gate.set()
    wait(client, job)


def test_switching_models_unloads_the_previous_one(server):
    client, backend, _ = server
    wait(client, submit(client))
    wait(client, submit(client, dict(REQ, model="m2")))
    assert backend.loaded == ["m1", "m2"] and backend.unloaded == 1


def test_idle_unload(tmp_path):
    backend, app = _serve(tmp_path, idle_unload_s=10.0)
    with TestClient(app) as client:
        wait(client, submit(client))
        assert app.state.reap_idle(now=time.monotonic() + 11) is True
        assert backend.unloaded == 1
        assert app.state.reap_idle(now=time.monotonic() + 11) is False


def test_mock_mode_never_loads_a_model(tmp_path):
    backend, app = _serve(tmp_path, mock=True)
    with TestClient(app) as client:
        st = wait(client, submit(client))
        assert st["frames"] == 2 and backend.loaded == []
        assert client.get("/health").json()["mock_mode"] is True
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_server.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'fxmotion_server'`.

- [ ] **Step 3: Write the module**

`server/fxmotion_server.py`:

```python
"""The HTTP contract every fxmotion model server speaks, in one module.

Spec: superpowers/specs/2026-09-29-fxhoudinimotion-design.md, section 4. A
model server is a Backend subclass plus `create_app(MyBackend(), ...)`. Jobs,
the request cache, progress, cancel and idle unloading live here, so each
model only says how to run itself. Times are seconds and positions Houdini
space everywhere in a request.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from fxmotion import clipformat

log = logging.getLogger("fxmotion_server")


class Segment(BaseModel):
    duration_s: float
    prompt: Optional[str] = None
    style: Optional[str] = None


class PathPoint(BaseModel):
    pos: list[float]  # [x, y, z], Houdini space
    time_s: Optional[float] = None  # all points timed, or none


class Keyframe(BaseModel):
    time_s: float
    world_pos: list[list[float]]  # (J, 3)
    world_rot: list[list[list[float]]]  # (J, 3, 3), KineFX row-vector
    joints: Optional[list[str]] = None  # None: whole body; names: effectors


class GenerateRequest(BaseModel):
    segments: list[Segment]
    root_path: Optional[list[PathPoint]] = None
    keyframes: Optional[list[Keyframe]] = None
    continue_from: Optional[dict] = None
    seed: Optional[int] = None
    model: str = ""
    force: bool = False  # bypass the cache
    options: dict = Field(default_factory=dict)  # backend-specific knobs


class JobStatus(BaseModel):
    job_id: str
    status: str  # queued | running | done | failed | cancelled
    prompt: Optional[str] = None
    frames: Optional[int] = None
    joints: Optional[int] = None
    error: Optional[str] = None
    elapsed: Optional[float] = None
    cached: Optional[bool] = None
    progress: Optional[float] = None
    phase: Optional[str] = None


class Progress:
    """Written by the backend on its worker thread, read by the event loop.
    Plain dict writes, no lock: each one is atomic in CPython."""

    def __init__(self, job: dict):
        self.job = job

    def set(self, fraction, phase=None, est_s=None, span=None) -> None:
        """`fraction` of the whole job done. `est_s` and `span` describe a
        phase that reports nothing while it runs (Kimodo's text encoding):
        it should take about est_s seconds and covers `span` of the job, so
        the displayed progress creeps across it instead of sitting still."""
        job = self.job
        if phase != job.get("phase"):
            job["phase_started"] = time.monotonic()
        job.update(
            progress=min(0.99, max(0.0, float(fraction))),
            phase=phase,
            est_s=est_s,
            span=span,
        )

    @property
    def cancelled(self) -> bool:
        return self.job.get("status") == "cancelled"


class Backend:
    """One model. Every method runs on a worker thread, one at a time."""

    name = "backend"
    skeleton = ""
    fps = 30.0
    models: tuple = ()
    default_model = ""
    # text, styles (list of names), root_path, keyframes, effectors,
    # continue, live
    capabilities: dict = {}

    def validate(self, req: GenerateRequest) -> None:
        """Raise ValueError for a request this model cannot run (422)."""

    def load(self, model: str) -> None:
        raise NotImplementedError

    def unload(self) -> None:
        raise NotImplementedError

    def generate(self, req: GenerateRequest, progress: Progress) -> dict:
        raise NotImplementedError

    def mock(self, req: GenerateRequest) -> dict:
        raise NotImplementedError

    def describe(self, req: GenerateRequest) -> str:
        return " | ".join(s.prompt or s.style or "" for s in req.segments)


def unsupported(req: GenerateRequest, backend: Backend) -> list:
    """What the request asks for that the backend's capabilities lack."""
    caps = backend.capabilities
    styles = caps.get("styles") or []
    out = []
    for i, seg in enumerate(req.segments, start=1):
        if seg.prompt is not None and not caps.get("text"):
            out.append("segment %d: text prompts" % i)
        if seg.style is not None:
            if not styles:
                out.append("segment %d: styles" % i)
            elif seg.style not in styles:
                out.append(
                    "segment %d: style %r (known: %s)"
                    % (i, seg.style, ", ".join(styles))
                )
    if req.root_path and not caps.get("root_path"):
        out.append("root_path")
    kinds = {"effectors" if kf.joints else "keyframes" for kf in req.keyframes or []}
    out += sorted(k for k in kinds if not caps.get(k))
    if req.continue_from and not caps.get("continue"):
        out.append("continue_from")
    return out


def _elapsed(job: dict) -> float:
    return round(time.monotonic() - job.get("started_at", time.monotonic()), 1)


def _display_progress(job: dict):
    prog, est = job.get("progress"), job.get("est_s")
    if prog is None or not est:
        return prog
    waited = time.monotonic() - job.get("phase_started", time.monotonic())
    span = job.get("span") or 0.0
    return min(0.99, prog + span * 0.9 * min(1.0, waited / est))


def create_app(
    backend: Backend,
    *,
    output_dir,
    mock: bool = False,
    idle_unload_s: float = 0.0,
    preload: bool = False,
) -> FastAPI:
    """The FastAPI app for one backend. `idle_unload_s` > 0 frees the model
    after that long without a job; `preload` loads the default model at
    startup (ignored in mock mode)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    jobs: dict = {}
    tasks: set = set()  # asyncio keeps only weak references to tasks
    lock = asyncio.Lock()  # one GPU, one generation at a time
    state = {"loaded": None, "last_used": time.monotonic()}

    def cache_key(req: GenerateRequest) -> str:
        payload = req.model_dump(exclude={"force"})
        payload["backend"] = backend.name
        blob = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()

    def ensure_loaded(model: str) -> None:
        model = model or backend.default_model
        if state["loaded"] != model:
            if state["loaded"] is not None:
                backend.unload()
            log.info("[LOAD] %s", model)
            backend.load(model)
            state["loaded"] = model
        state["last_used"] = time.monotonic()

    def reap_idle(now=None) -> bool:
        now = time.monotonic() if now is None else now
        if (
            idle_unload_s > 0
            and state["loaded"] is not None
            and not lock.locked()
            and now - state["last_used"] >= idle_unload_s
        ):
            log.info("[UNLOAD] %s after %.0f s idle", state["loaded"], idle_unload_s)
            backend.unload()
            state["loaded"] = None
            return True
        return False

    async def reaper():
        while True:
            await asyncio.sleep(max(1.0, idle_unload_s / 4))
            reap_idle()

    @asynccontextmanager
    async def lifespan(app):
        if preload and not mock:
            await asyncio.to_thread(ensure_loaded, "")
        task = asyncio.create_task(reaper()) if idle_unload_s > 0 else None
        yield
        if task is not None:
            task.cancel()

    app = FastAPI(title="fxmotion %s server" % backend.name, lifespan=lifespan)
    app.state.reap_idle = reap_idle

    def produce(req: GenerateRequest, job: dict) -> dict:
        if mock:
            return backend.mock(req)
        ensure_loaded(req.model)
        try:
            return backend.generate(req, Progress(job))
        finally:
            state["last_used"] = time.monotonic()

    def finish(job: dict, path: Path, cached: bool) -> None:
        with np.load(path) as z:
            frames, joints = z["world_pos"].shape[:2]
        job.update(
            status="done",
            path=str(path),
            frames=int(frames),
            joints=int(joints),
            cached=cached,
            elapsed=_elapsed(job),
            progress=1.0,
            phase=None,
            est_s=None,
        )

    async def run(job_id: str, req: GenerateRequest) -> None:
        job = jobs[job_id]
        if job["status"] == "cancelled":
            return
        job.update(status="running", progress=0.0)
        path = output_dir / ("%s.npz" % cache_key(req))
        try:
            if not req.force and path.exists():
                finish(job, path, cached=True)
                return
            async with lock:
                if job["status"] == "cancelled":
                    return
                log.info("[GEN] %s %s", job_id[:8], backend.describe(req))
                clip = await asyncio.to_thread(produce, req, job)
            if job["status"] == "cancelled":  # discard the result
                return
            await asyncio.to_thread(clipformat.save, path, clip)
            finish(job, path, cached=False)
        except Exception as exc:  # never leave a job stuck in "running"
            if job.get("status") != "cancelled":
                log.exception("[FAIL] %s", job_id[:8])
                job.update(
                    status="failed", error=str(exc)[-500:], elapsed=_elapsed(job)
                )

    @app.get("/health")
    def health() -> dict:
        return {
            "backend": backend.name,
            "skeleton": backend.skeleton,
            "fps": backend.fps,
            "models": list(backend.models),
            "default_model": backend.default_model,
            "capabilities": backend.capabilities,
            "mock_mode": mock,
            "loaded_model": state["loaded"],
        }

    @app.post("/generate", status_code=202)
    async def generate(req: GenerateRequest) -> JobStatus:
        missing = unsupported(req, backend)
        if missing:
            raise HTTPException(
                422, "%s: not supported: %s" % (backend.name, "; ".join(missing))
            )
        try:
            backend.validate(req)
        except ValueError as e:
            raise HTTPException(422, str(e)) from e
        job_id = uuid.uuid4().hex
        desc = backend.describe(req)
        jobs[job_id] = {"status": "queued", "started_at": time.monotonic(), "prompt": desc}
        task = asyncio.create_task(run(job_id, req))
        tasks.add(task)
        task.add_done_callback(tasks.discard)
        return JobStatus(job_id=job_id, status="queued", prompt=desc)

    def status(job_id: str) -> JobStatus:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "Job %s not found." % job_id)
        elapsed = job.get("elapsed")
        if elapsed is None:
            elapsed = _elapsed(job)
        return JobStatus(
            job_id=job_id,
            status=job["status"],
            prompt=job.get("prompt"),
            frames=job.get("frames"),
            joints=job.get("joints"),
            error=job.get("error"),
            elapsed=elapsed,
            cached=job.get("cached"),
            progress=_display_progress(job),
            phase=job.get("phase"),
        )

    @app.get("/jobs/{job_id}")
    def job_status(job_id: str) -> JobStatus:
        return status(job_id)

    @app.get("/jobs/{job_id}/download")
    def download(job_id: str):
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "Job %s not found." % job_id)
        if job["status"] != "done":
            raise HTTPException(409, "Job %s is %s." % (job_id, job["status"]))
        return FileResponse(
            job["path"],
            media_type="application/octet-stream",
            filename="%s.npz" % job_id,
        )

    @app.post("/jobs/{job_id}/cancel")
    async def cancel(job_id: str) -> JobStatus:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "Job %s not found." % job_id)
        if job["status"] in ("queued", "running"):
            # in-process inference cannot be interrupted: a queued job is
            # skipped and a running one's result discarded
            job.update(status="cancelled", elapsed=_elapsed(job))
            log.info("[CANCEL] %s", job_id[:8])
        return status(job_id)

    return app
```

In `.github/workflows/tests.yml`, replace the `Install pytest` step with:

```yaml
      - name: Install test dependencies
        # numpy for the clip format, fastapi + httpx for the server tests
        # (TestClient), requests for fxmotion.client.
        run: pip install pytest numpy requests fastapi httpx
```

Append `"fxmotion_server"` to `PURE` in `tests/test_imports.py`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_server.py tests/test_imports.py -q`
Expected: all pass.

- [ ] **Step 5: Lint and commit**

```bash
uvx ruff@0.16.7 check --fix . && uvx ruff@0.16.7 format . && uvx ruff@0.16.7 check --fix . && uvx ruff@0.16.7 format . && uvx ruff@0.16.7 check . && uvx ruff@0.16.7 format --check .
git add server/fxmotion_server.py tests/test_server.py tests/test_imports.py .github/workflows/tests.yml
git commit -m "feat(server): add the shared model server every backend runs on

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: the Kimodo backend

**Files:**
- Create: `server/kimodo_backend.py`
- Delete: `kimodo_server.py`
- Modify: `docker-compose.bridge.yaml`, `tests/test_imports.py`
- Test: `tests/test_kimodo_backend.py`

**Interfaces:**
- Consumes: `fxmotion_server.Backend`, `create_app`, `Progress`; `kimodo_adapter.kimodo_inputs`, `to_clip`, `FPS`.
- Produces (module `kimodo_backend`): `MODELS = ("Kimodo-SOMA-RP-v1.1", "Kimodo-SOMA-SEED-v1.1", "Kimodo-SOMA-RP-v1")`; `class KimodoBackend(fs.Backend)` with `__init__(default_model="", mock_clip=None, encode_est_s=30.0)` and capabilities `{"text": True, "styles": [], "root_path": True, "keyframes": True, "effectors": True, "continue": True, "live": False}`; `class _Sampler(backend, progress, expected)` (a tqdm stand-in); `make_app() -> FastAPI` reading `OUTPUT_DIR`, `MOCK_MODE`, `KIMODO_MODEL`, `FXMOTION_MOCK_CLIP`, `KIMODO_ENCODE_EST_S`, `FXMOTION_IDLE_UNLOAD_S`.

- [ ] **Step 1: Write the failing tests**

`tests/test_kimodo_backend.py`:

```python
"""The Kimodo server end to end in MOCK_MODE (no torch, no GPU), plus its
progress stand-in."""

import os
import subprocess
import sys
import time
from pathlib import Path

import fxmotion_server as fs
import kimodo_backend as kb
import numpy as np
from fastapi.testclient import TestClient

from fxmotion import clipformat

HERE = Path(__file__).resolve().parent
FIXTURE = HERE / "fixtures" / "kimodo_stop.npz"


def _client(tmp_path):
    app = fs.create_app(kb.KimodoBackend(mock_clip=FIXTURE), output_dir=tmp_path, mock=True)
    return TestClient(app)


def _wait(client, job_id):
    end = time.monotonic() + 5
    while True:
        st = client.get("/jobs/%s" % job_id).json()
        if st["status"] in ("done", "failed", "cancelled"):
            return st
        assert time.monotonic() < end
        time.sleep(0.02)


def test_mock_generate_returns_a_soma77_clip_on_the_path(tmp_path):
    req = {
        "segments": [{"prompt": "a person walks", "duration_s": 1.5}],
        "root_path": [{"pos": [1.0, 0.0, 2.0]}, {"pos": [1.0, 0.0, 5.0]}],
    }
    with _client(tmp_path) as client:
        caps = client.get("/health").json()["capabilities"]
        assert caps["keyframes"] and caps["continue"] and not caps["live"]
        st = _wait(client, client.post("/generate", json=req).json()["job_id"])
        assert st["status"] == "done", st
        out = tmp_path / "got.npz"
        out.write_bytes(client.get("/jobs/%s/download" % st["job_id"]).content)
    clip = clipformat.load(out)
    assert str(clip["skeleton"]) == "soma77"
    src = clipformat.meta(clip, "source")
    assert src["canon"] == [1.0, 2.0, 0.0] and src["backend"] == "kimodo"
    with np.load(FIXTURE) as z:
        root = z["posed_joints"][0, 0]
    assert np.allclose(clip["world_pos"][0, 0], root + [1.0, 0.0, 2.0], atol=1e-5)


def test_kimodo_refuses_styles_and_empty_prompts(tmp_path):
    with _client(tmp_path) as client:
        r = client.post("/generate", json={"segments": [{"style": "walk", "duration_s": 1.0}]})
        assert r.status_code == 422
        assert r.json()["detail"] == "kimodo: not supported: segment 1: styles"
        r = client.post("/generate", json={"segments": [{"prompt": " ", "duration_s": 1.0}]})
        assert r.status_code == 422
        assert r.json()["detail"] == "kimodo: segment 1 has no prompt"


class _Recorder:
    def __init__(self):
        self.calls = []

    def set(self, fraction, phase=None, est_s=None, span=None):
        self.calls.append((round(fraction, 3), phase, est_s))


def test_sampler_walks_encode_denoise_post_and_learns_the_encode_time():
    backend = kb.KimodoBackend(encode_est_s=30.0)
    rec = _Recorder()
    sampler = kb._Sampler(backend, rec, expected=2)
    list(sampler(range(4)))
    list(sampler(range(4)))
    phases = [p for _, p, _ in rec.calls]
    assert phases[0] == "encoding text"
    assert "denoising segment 1/2" in phases and "denoising segment 2/2" in phases
    assert phases[-1] == "post-processing"
    assert backend.encode_est_s < 30.0  # replaced by the measured encode


def test_module_imports_and_builds_without_torch(tmp_path):
    env = dict(os.environ)
    env.update(
        PYTHONPATH=os.pathsep.join([str(HERE.parent / "houdini" / "python"), str(HERE.parent / "server")]),
        OUTPUT_DIR=str(tmp_path),
        MOCK_MODE="1",
    )
    code = (
        "import sys; sys.modules['torch'] = None; sys.modules['kimodo'] = None; "
        "import kimodo_backend; kimodo_backend.make_app()"
    )
    subprocess.run([sys.executable, "-c", code], check=True, env=env)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_kimodo_backend.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'kimodo_backend'`.

- [ ] **Step 3: Write the backend**

Create `server/kimodo_backend.py`. Move `_build_initial_motion` and `_build_constraints` **unchanged** from `kimodo_server.py` (lines 64-157, including their docstrings) into it where marked; the rest is new:

```python
"""The Kimodo model server: inference on the shared fxmotion server.

Runs in the kimodo:1.0 image (docker-compose.bridge.yaml), which provides
torch and the kimodo package. Everything that needs them is imported inside a
method, so this module imports, and the MOCK_MODE server runs, without them.
The pure conversions live in kimodo_adapter.py.

    uvicorn kimodo_backend:make_app --factory --host 0.0.0.0 --port 8001
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import fxmotion_server as fs
import kimodo_adapter as ka
import numpy as np

log = logging.getLogger("kimodo_backend")
logging.basicConfig(level=logging.INFO)

MODELS = ("Kimodo-SOMA-RP-v1.1", "Kimodo-SOMA-SEED-v1.1", "Kimodo-SOMA-RP-v1")


# _build_initial_motion(cf, model): moved unchanged from kimodo_server.py
# _build_constraints(constraints, model): moved unchanged from kimodo_server.py


class _Sampler:
    """Stand-in for tqdm: Kimodo wraps one denoising loop per segment with
    progress_bar(indices). A segment costs roughly 30 s of text encoding (CPU,
    reports nothing), ~8 s of denoising (GPU) and a few seconds of
    post-processing, so the fraction is (finished loops + position in the
    current loop) / segments, and the encode phase is announced with an
    estimate the server's display creeps across. The first measured encode
    replaces the estimate for every later job."""

    def __init__(self, backend, progress, expected: int):
        self.backend, self.progress = backend, progress
        self.expected, self.done = max(1, expected), 0
        self._encoding_since = None
        self._encode()

    def _encode(self) -> None:
        self._encoding_since = time.monotonic()
        self.progress.set(
            self.done / self.expected,
            "encoding text",
            est_s=self.backend.encode_est_s,
            span=1.0 / self.expected,
        )

    def __call__(self, iterable, **_):
        if self._encoding_since is not None:
            took = time.monotonic() - self._encoding_since
            self.backend.encode_est_s = max(0.1, took)
            self._encoding_since = None
        items = list(iterable)
        n = max(1, len(items))
        phase = "denoising segment %d/%d" % (self.done + 1, self.expected)
        for i, item in enumerate(items):
            self.progress.set((self.done + i / n) / self.expected, phase)
            yield item
        self.done += 1
        if self.done >= self.expected:
            self.progress.set(1.0, "post-processing")
        else:
            self._encode()


def _segments(inp: ka.KimodoInputs) -> list:
    out, start = [], 0
    for text, n in zip(inp.texts, inp.num_frames):
        out.append({"prompt": text, "start": start, "end": start + n - 1})
        start += n
    return out


class KimodoBackend(fs.Backend):
    name = "kimodo"
    skeleton = "soma77"
    fps = ka.FPS
    models = MODELS
    capabilities = {
        "text": True,
        "styles": [],
        "root_path": True,
        "keyframes": True,
        "effectors": True,
        "continue": True,
        "live": False,
    }

    def __init__(self, default_model="", mock_clip=None, encode_est_s=30.0):
        self.default_model = default_model or MODELS[0]
        self.mock_clip = Path(mock_clip) if mock_clip else None
        self.encode_est_s = float(encode_est_s)
        self._model = None

    def validate(self, req) -> None:
        ka.kimodo_inputs(req.model_dump())  # AdapterError is a ValueError

    def load(self, model: str) -> None:
        import torch
        from kimodo import load_model
        from kimodo.model.registry import resolve_model_name

        key = resolve_model_name(model or "", default_family="Kimodo")
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        log.info("[RESIDENT] loading model %s on %s ...", key, device)
        self._model, resolved = load_model(
            key, device=device, default_family="Kimodo", return_resolved_name=True
        )
        log.info("[RESIDENT] model ready: %s", resolved)

    def unload(self) -> None:
        self._model = None
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def generate(self, req, progress) -> dict:
        inp = ka.kimodo_inputs(req.model_dump())
        model = self._model
        if req.seed is not None:
            import torch

            torch.manual_seed(int(req.seed))
        constraint_lst = _build_constraints(inp.constraints, model)
        initial_motion = _build_initial_motion(inp.continue_from, model)
        sampler = _Sampler(self, progress, len(inp.texts))
        # Kimodo's multi-prompt path does not forward progress_bar to the
        # sampling loop (kimodo_model._multiprompt calls self._generate
        # without it), so inject it; inference is serialised by the server,
        # so patching the resident model is safe.
        orig = model._generate

        def _with_progress(*a, **k):
            k.setdefault("progress_bar", sampler)
            return orig(*a, **k)

        model._generate = _with_progress
        try:
            output = model(
                inp.texts,
                inp.num_frames,
                num_denoising_steps=100,
                num_samples=1,
                multi_prompt=True,
                num_transition_frames=inp.transition_frames,
                initial_motion=initial_motion,
                post_processing=True,
                constraint_lst=constraint_lst,
                return_numpy=True,
                progress_bar=sampler,
            )
        finally:
            del model._generate  # back to the class method
        n = int(output["posed_joints"].shape[0])
        single = {
            k: (v[0] if hasattr(v, "shape") and len(v.shape) > 0 and v.shape[0] == n else v)
            for k, v in output.items()
        }
        return ka.to_clip(
            single,
            inp.canon,
            segments=_segments(inp),
            source={"model": req.model or self.default_model, "seed": req.seed},
        )

    def mock(self, req) -> dict:
        inp = ka.kimodo_inputs(req.model_dump())
        if self.mock_clip is None or not self.mock_clip.exists():
            raise FileNotFoundError(
                "MOCK_MODE serves a Kimodo NPZ, none found at %s" % self.mock_clip
            )
        with np.load(self.mock_clip) as z:
            npz = {k: z[k] for k in z.files}
        return ka.to_clip(
            npz, inp.canon, segments=_segments(inp), source={"model": "mock", "seed": req.seed}
        )


def make_app():
    out = Path(os.environ.get("OUTPUT_DIR", "/workspace/output"))
    backend = KimodoBackend(
        default_model=os.environ.get("KIMODO_MODEL", ""),
        mock_clip=os.environ.get("FXMOTION_MOCK_CLIP", str(out / "dev_reference.npz")),
        encode_est_s=float(os.environ.get("KIMODO_ENCODE_EST_S", "30")),
    )
    return fs.create_app(
        backend,
        output_dir=out,
        mock=os.environ.get("MOCK_MODE", "1") == "1",
        idle_unload_s=float(os.environ.get("FXMOTION_IDLE_UNLOAD_S", "0")),
        preload=True,
    )
```

Then delete the old server:

```bash
git rm kimodo_server.py
```

Append `"kimodo_backend"` to `PURE` in `tests/test_imports.py`.

- [ ] **Step 4: Update the compose file**

In `docker-compose.bridge.yaml`:
1. Add a third line to the `x-workspace` list: `  - ${FXMOTION_ROOT:?set FXMOTION_ROOT to your fxhoudinimotion clone}:/fxmotion:ro`.
2. Replace the api `command:` with `command: uvicorn kimodo_backend:make_app --factory --host 0.0.0.0 --port ${KIMODO_PORT:-8001}`.
3. Add to the api `environment:`:

```yaml
      # The server code is mounted from the fxhoudinimotion clone, not copied.
      PYTHONPATH: /fxmotion/server:/fxmotion/houdini/python
      # Free the model's VRAM after this many idle seconds (0 = keep it).
      FXMOTION_IDLE_UNLOAD_S: ${FXMOTION_IDLE_UNLOAD_S:-0}
```

4. In the header comment, replace "The kimodo_motion HDA fetches the NPZ over HTTP" with "The kimodo_motion HDA fetches the clip over HTTP", and add after the usage lines: `# FXMOTION_ROOT must point at the fxhoudinimotion clone; the server code is mounted from it.`

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python -m pytest -q`
Expected: all pass (the whole suite, since `kimodo_server.py` is gone).

- [ ] **Step 6: Lint and commit**

```bash
uvx ruff@0.16.7 check --fix . && uvx ruff@0.16.7 format . && uvx ruff@0.16.7 check --fix . && uvx ruff@0.16.7 format . && uvx ruff@0.16.7 check . && uvx ruff@0.16.7 format --check .
git add server/kimodo_backend.py tests/test_kimodo_backend.py tests/test_imports.py docker-compose.bridge.yaml
git commit -m "feat(server): run Kimodo on the shared server, mounted from the repo

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: the Houdini clip loader

**Files:**
- Create: `houdini/python/fxmotion/clip.py`
- Modify: `tests/test_imports.py`
- Test: `tests/test_clip.py`

**Interfaces:**
- Consumes: `clipformat.load`, `clipformat.meta`, `fxmotion.skeletons.Skeleton`.
- Produces: `clip.load(path) -> dict` (LRU of 8 by path + mtime), `clip.sample_index(scene_frame, start_frame, scene_fps, clip_fps, retime, frames) -> int`, `clip.joint_paths(names, parents) -> list[str]`, `clip.joint_frames(clip, index) -> (pos (J,3), xform (J,9), local (J,16))`, `clip.build(geo, clip, index)`, `clip.build_rest(geo, skeleton)`. Point attributes written by `build`: `name`, `path`, `parent_id`, `transform`, `localtransform`, optional `contact`; detail attributes `fxmotion_skeleton` (str), `fxmotion_fps` (float), `fxmotion_source` (JSON str).

- [ ] **Step 1: Write the failing tests**

`tests/test_clip.py`:

```python
"""Clip -> KineFX math, against the frozen 1.1 cook."""

import os
from pathlib import Path

import _reference_v11 as v11
import kimodo_adapter as ka
import numpy as np
import pytest
from cliptools import tiny

from fxmotion import clip, clipformat

FIXTURE = Path(__file__).parent / "fixtures" / "kimodo_stop.npz"


def _npz():
    with np.load(FIXTURE) as z:
        return {k: z[k] for k in z.files}


@pytest.mark.parametrize("canon", (ka.Canon(), ka.Canon(1.5, -2.0, 0.7)))
def test_joint_frames_match_the_v11_cook(canon):
    npz = _npz()
    c = ka.to_clip(npz, canon)
    for frame in (0, 17, 49):
        pos, xform, local = clip.joint_frames(c, frame)
        rpos, rxform, rlocal = v11.cook(npz, frame, canon.ox, canon.oz, canon.ang)
        assert np.abs(pos - rpos).max() < 1e-5
        assert np.abs(xform - rxform).max() < 1e-5
        assert np.abs(local - rlocal).max() < 1e-4


def test_sample_index_matches_v11():
    # 1.1: f = frame - start; with retime f *= clip_fps / scene_fps; round; clamp
    assert clip.sample_index(1, 1, 24, 30, True, 50) == 0
    assert clip.sample_index(25, 1, 24, 30, True, 50) == 30
    assert clip.sample_index(25, 1, 24, 30, False, 50) == 24
    assert clip.sample_index(1001, 1001, 25, 30, True, 50) == 0
    assert clip.sample_index(1011, 1001, 25, 30, True, 50) == 12
    assert clip.sample_index(-5, 1, 24, 30, True, 50) == 0
    assert clip.sample_index(500, 1, 24, 30, True, 50) == 49


def test_joint_paths():
    paths = clip.joint_paths(["Hips", "Spine1", "LeftUpLeg"], [-1, 0, 0])
    assert paths == ["/Hips", "/Hips/Spine1", "/Hips/LeftUpLeg"]


def test_load_caches_by_file_version(tmp_path):
    a, b = tmp_path / "a.npz", tmp_path / "b.npz"
    clipformat.save(a, tiny(3))
    clipformat.save(b, tiny(4))
    first = clip.load(a)
    assert clip.load(b) is not first
    assert clip.load(a) is first  # two nodes alternating stay cached
    clipformat.save(a, tiny(5))
    st = a.stat()
    os.utime(a, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
    assert clipformat.frame_count(clip.load(a)) == 5


def test_load_names_what_is_wrong_with_a_native_npz():
    with pytest.raises(clipformat.ClipError, match="missing format"):
        clip.load(FIXTURE)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_clip.py -q`
Expected: FAIL with `ImportError: cannot import name 'clip' from 'fxmotion'`.

- [ ] **Step 3: Write the module**

`houdini/python/fxmotion/clip.py`:

```python
"""A clip on disk -> KineFX skeleton geometry.

The math is pure and tested offline against the 1.1 cook; build() and
build_rest() take a hou.Geometry and run inside the generator SOPs. They set
attributes in batches (one HOM call per attribute, not per point).
"""

from __future__ import annotations

import json
import os

import numpy as np

from . import clipformat

_CACHE: dict = {}
_CACHE_SIZE = 8  # generator nodes in one scene, each with its own clip


def load(path) -> dict:
    """Every array of a clip, read once per file version: the animated SOP
    cooks on every frame change and an NpzFile re-decompresses per lookup."""
    key = (str(path), os.path.getmtime(path))
    hit = _CACHE.get(key)
    if hit is None:
        if len(_CACHE) >= _CACHE_SIZE:
            _CACHE.pop(next(iter(_CACHE)))  # oldest first
        hit = _CACHE[key] = clipformat.load(path)
    return hit


def sample_index(scene_frame, start_frame, scene_fps, clip_fps, retime, frames) -> int:
    """Which clip sample a scene frame shows. The clip starts on Start
    Frame; with Retime on, scene frames map onto samples by real time, so a
    3 s clip lasts 3 s at any $FPS. The first sample holds before the start,
    the last after the end. Nearest sample, no blending."""
    f = float(scene_frame) - float(start_frame)
    if retime:
        f = f * float(clip_fps) / float(scene_fps)
    return max(0, min(int(round(f)), int(frames) - 1))


def joint_paths(names, parents) -> list:
    out = []
    for name, parent in zip(names, parents):
        out.append("/" + name if parent < 0 else out[parent] + "/" + name)
    return out


def joint_frames(clip, index):
    """World positions (J, 3), KineFX transforms (J, 9) and local 4x4s
    (J, 16) at one sample. Row-vector: local = world @ inverse(parent)."""
    pos = clip["world_pos"][index].astype(np.float64)
    rot = clip["world_rot"][index].astype(np.float64)
    count = len(pos)
    world = np.zeros((count, 4, 4))
    world[:, :3, :3] = rot
    world[:, 3, :3] = pos
    world[:, 3, 3] = 1.0
    parents = np.asarray(clip["parents"])
    local = world.copy()
    has = parents >= 0
    local[has] = world[has] @ np.linalg.inv(world[parents[has]])
    return pos, rot.reshape(count, 9), local.reshape(count, 16)


def _skeleton_points(geo, pos, names, parents) -> None:
    import hou

    geo.addAttrib(hou.attribType.Point, "name", "")
    geo.addAttrib(hou.attribType.Point, "path", "")
    geo.addAttrib(hou.attribType.Point, "parent_id", -1)
    geo.createPoints([hou.Vector3(p) for p in np.asarray(pos).tolist()])
    geo.setPointStringAttribValues("name", tuple(names))
    geo.setPointStringAttribValues("path", tuple(joint_paths(names, parents)))
    geo.setPointIntAttribValues("parent_id", tuple(parents))
    bones = tuple((p, i) for i, p in enumerate(parents) if p >= 0)
    geo.createPolygons(bones, False)  # is_closed only works positionally


def _detail(geo, name, value) -> None:
    import hou

    geo.addAttrib(hou.attribType.Global, name, "" if isinstance(value, str) else 0.0)
    geo.setGlobalAttribValue(name, value)


def build(geo, clip, index) -> None:
    """The animated skeleton at one sample, plus the fxmotion_* details."""
    import hou

    pos, xform, local = joint_frames(clip, index)
    names = [str(n) for n in clip["joint_names"]]
    parents = [int(p) for p in clip["parents"]]
    _skeleton_points(geo, pos, names, parents)
    # HOM wants Python floats, not numpy scalars: hence tolist() throughout
    geo.addAttrib(hou.attribType.Point, "transform", tuple(np.eye(3).flatten().tolist()))
    geo.addAttrib(hou.attribType.Point, "localtransform", tuple(np.eye(4).flatten().tolist()))
    geo.setPointFloatAttribValues("transform", tuple(xform.flatten().tolist()))
    geo.setPointFloatAttribValues("localtransform", tuple(local.flatten().tolist()))
    if "contacts" in clip:
        geo.addAttrib(hou.attribType.Point, "contact", 0)
        geo.setPointIntAttribValues("contact", tuple(int(v) for v in clip["contacts"][index]))
    _detail(geo, "fxmotion_skeleton", str(clip["skeleton"]))
    _detail(geo, "fxmotion_fps", float(clip["fps"]))
    source = clipformat.meta(clip, "source") or {}
    _detail(geo, "fxmotion_source", json.dumps(source, sort_keys=True))


def build_rest(geo, skeleton) -> None:
    """The skeleton's rest pose (the T-Pose output)."""
    import hou

    names, parents = list(skeleton.joint_names), list(skeleton.parents)
    _skeleton_points(geo, skeleton.rest_pos, names, parents)
    geo.addAttrib(hou.attribType.Point, "transform", tuple(np.eye(3).flatten().tolist()))
    geo.setPointFloatAttribValues("transform", tuple(skeleton.rest_rot.flatten().tolist()))
    _detail(geo, "fxmotion_skeleton", skeleton.name)
```

Append `"fxmotion.clip"` to `PURE` in `tests/test_imports.py`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_clip.py tests/test_imports.py -q`
Expected: all pass.

- [ ] **Step 5: Lint and commit**

```bash
uvx ruff@0.16.7 check --fix . && uvx ruff@0.16.7 format . && uvx ruff@0.16.7 check --fix . && uvx ruff@0.16.7 format . && uvx ruff@0.16.7 check . && uvx ruff@0.16.7 format --check .
git add houdini/python/fxmotion/clip.py tests/test_clip.py tests/test_imports.py
git commit -m "feat(fxmotion): load clips into KineFX skeletons, matching the 1.1 cook

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: the HTTP client and path helpers

**Files:**
- Create: `houdini/python/fxmotion/client.py`, `houdini/python/fxmotion/paths.py`
- Modify: `tests/test_imports.py`
- Test: `tests/test_client.py`, `tests/test_paths.py`

**Interfaces:**
- Produces: `client.ServerError(message, status=None)` (attribute `status: int | None`), `client.Rejected(ServerError)`, `client.health(url, session=requests) -> dict`, `client.submit(url, payload, session=requests) -> str`, `client.job(url, job_id, session=requests) -> dict`, `client.cancel(url, job_id, session=requests) -> dict`, `client.download(url, job_id, dest_dir, session=requests) -> str` (forward-slash path to `<dest_dir>/<job_id>.npz`). `paths.thin(xz, count) -> list[int]`, `paths.scene_seconds(scene_frame, start_frame, scene_fps, clip_fps, retime) -> float`.

- [ ] **Step 1: Write the failing tests**

`tests/test_client.py`:

```python
"""fxmotion.client against a fake requests session: no network."""

import pytest
import requests

from fxmotion import client


class FakeResponse:
    def __init__(self, status=200, body=None, content=b"", reason="OK"):
        self.status_code, self._body, self.content = status, body, content
        self.reason, self.text = reason, str(body)

    def json(self):
        if self._body is None:
            raise ValueError("no json")
        return self._body

    def iter_content(self, chunk_size=1):
        yield self.content

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeSession:
    def __init__(self, response=None, error=None):
        self.response, self.error, self.calls = response, error, []

    def _do(self, method, url, **kw):
        self.calls.append((method, url, kw))
        if self.error:
            raise self.error
        return self.response

    def get(self, url, **kw):
        return self._do("GET", url, **kw)

    def post(self, url, **kw):
        return self._do("POST", url, **kw)


def test_health_strips_the_trailing_slash():
    s = FakeSession(FakeResponse(body={"backend": "kimodo"}))
    assert client.health("http://h:8001/", session=s) == {"backend": "kimodo"}
    assert s.calls[0][1] == "http://h:8001/health"


def test_unreachable_server_is_a_server_error():
    s = FakeSession(error=requests.ConnectionError("refused"))
    with pytest.raises(client.ServerError, match="unreachable at http://h"):
        client.health("http://h", session=s)


def test_a_422_is_rejected_with_the_servers_words():
    body = {"detail": "kimodo: segment 1 has no prompt"}
    s = FakeSession(FakeResponse(422, body, reason="Unprocessable"))
    with pytest.raises(client.Rejected, match="segment 1 has no prompt"):
        client.submit("http://h", {"segments": []}, session=s)


def test_submit_returns_the_job_id():
    s = FakeSession(FakeResponse(202, {"job_id": "abc", "status": "queued"}))
    assert client.submit("http://h", {"x": 1}, session=s) == "abc"
    assert s.calls[0][2]["json"] == {"x": 1}


def test_a_lost_job_carries_its_status():
    s = FakeSession(FakeResponse(404, {"detail": "Job x not found."}, reason="Not Found"))
    with pytest.raises(client.ServerError) as e:
        client.job("http://h", "x", session=s)
    assert e.value.status == 404


def test_download_creates_the_folder(tmp_path):
    s = FakeSession(FakeResponse(content=b"NPZBYTES"))
    dest = tmp_path / "not" / "there"
    out = client.download("http://h", "job1", str(dest), session=s)
    assert out == (dest / "job1.npz").as_posix()
    assert (dest / "job1.npz").read_bytes() == b"NPZBYTES"
```

`tests/test_paths.py`:

```python
"""Root-path thinning and scene time, against 1.1's behaviour."""

from fxmotion import paths


def test_thin_keeps_everything_when_asked_for_as_many_or_fewer_than_two():
    xz = [[float(i), 0.0] for i in range(5)]
    assert paths.thin(xz, 0) == [0, 1, 2, 3, 4]
    assert paths.thin(xz, 8) == [0, 1, 2, 3, 4]


def test_thin_by_arc_length_keeps_first_and_last():
    xz = [[float(i), 0.0] for i in range(11)]
    assert paths.thin(xz, 3) == [0, 4, 10]
    assert paths.thin(xz, 2) == [0, 10]


def test_scene_seconds_matches_v11_samples():
    # 1.1 to_sample: retime -> round((f - start) * clip_fps / scene_fps),
    # else int(f - start). 2.0 sends seconds and the server multiplies by 30,
    # so the sample *value* must match; compared before rounding, because
    # rounding two float spellings of an exact .5 can land either side.
    for scene_fps in (24, 25, 30):
        for f in range(1, 80):
            t = paths.scene_seconds(f, 1, scene_fps, 30, True)
            assert abs(t * 30 - (f - 1) * 30 / scene_fps) < 1e-9
            t = paths.scene_seconds(f, 1, scene_fps, 30, False)
            assert abs(t * 30 - (f - 1)) < 1e-9
    assert paths.scene_seconds(0, 10, 24, 30, True) == 0.0  # before the start
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_client.py tests/test_paths.py -q`
Expected: FAIL with `ImportError: cannot import name 'client' from 'fxmotion'`.

- [ ] **Step 3: Write the modules**

`houdini/python/fxmotion/client.py`:

```python
"""HTTP to an fxmotion model server. No hou; the session is injectable so
the tests need no network."""

from __future__ import annotations

from pathlib import Path

import requests


class ServerError(RuntimeError):
    """The server is unreachable or answered with an error."""

    def __init__(self, message, status=None):
        super().__init__(message)
        self.status = status


class Rejected(ServerError):
    """422: the server understood the request and refuses it."""


def _base(url: str) -> str:
    return url.rstrip("/")


def _check(resp):
    if resp.status_code == 422:
        try:
            detail = resp.json().get("detail")
        except ValueError:
            detail = resp.text
        raise Rejected(detail if isinstance(detail, str) else str(detail), 422)
    if resp.status_code >= 400:
        raise ServerError(
            "%s %s: %s" % (resp.status_code, resp.reason, resp.text[:300]),
            resp.status_code,
        )
    return resp


def _call(method, url, **kw):
    try:
        return _check(method(url, **kw))
    except requests.RequestException as e:
        raise ServerError("server unreachable at %s: %s" % (url, e)) from e


def health(url, session=requests) -> dict:
    return _call(session.get, _base(url) + "/health", timeout=5).json()


def submit(url, payload, session=requests) -> str:
    resp = _call(session.post, _base(url) + "/generate", json=payload, timeout=60)
    return resp.json()["job_id"]


def job(url, job_id, session=requests) -> dict:
    return _call(session.get, "%s/jobs/%s" % (_base(url), job_id), timeout=10).json()


def cancel(url, job_id, session=requests) -> dict:
    return _call(session.post, "%s/jobs/%s/cancel" % (_base(url), job_id), timeout=10).json()


def download(url, job_id, dest_dir, session=requests) -> str:
    """Stream the job's clip to <dest_dir>/<job_id>.npz; returns that path
    with forward slashes, Houdini's own convention."""
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    out = dest / ("%s.npz" % job_id)
    target = "%s/jobs/%s/download" % (_base(url), job_id)
    try:
        with session.get(target, timeout=120, stream=True) as resp:
            _check(resp)
            with open(out, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    fh.write(chunk)
    except requests.RequestException as e:
        raise ServerError("download failed from %s: %s" % (target, e)) from e
    return out.as_posix()
```

`houdini/python/fxmotion/paths.py`:

```python
"""Root-path and timing helpers for the generator nodes. Pure."""

from __future__ import annotations

import math


def thin(xz, count) -> list:
    """Indices of about `count` points spread by arc length, first and last
    always kept. A waypoint every frame or two pins the pelvis to a
    constant-speed glide; a handful of anchors lets the model put its own
    stride rhythm back between them. count < 2 keeps every point."""
    if count < 2 or len(xz) <= count:
        return list(range(len(xz)))
    cum = [0.0]
    for (ax, az), (bx, bz) in zip(xz, xz[1:]):
        cum.append(cum[-1] + math.hypot(bx - ax, bz - az))
    total = cum[-1] or 1.0
    keep, j = [], 0
    for k in range(count):
        target = total * k / (count - 1)
        while j < len(cum) - 1 and cum[j + 1] < target:
            j += 1
        keep.append(j if k < count - 1 else len(xz) - 1)
    return sorted(set(keep))


def scene_seconds(scene_frame, start_frame, scene_fps, clip_fps, retime) -> float:
    """Clip time, in seconds, that a scene frame lands on: what a request
    speaks. Retime on maps scene frames to real time; off, one clip sample
    per scene frame, so the model's own rate converts."""
    f = float(scene_frame) - float(start_frame)
    return max(0.0, f / float(scene_fps) if retime else f / float(clip_fps))
```

Append `"fxmotion.client"` and `"fxmotion.paths"` to `PURE` in `tests/test_imports.py`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_client.py tests/test_paths.py tests/test_imports.py -q`
Expected: all pass. If `test_thin_by_arc_length_keeps_first_and_last` fails on `[0, 4, 10]`, trace 1.1's loop by hand (target 5.0: `j` advances while `cum[j + 1] < 5.0`, stopping at `j = 4` because `cum[5] = 5.0` is not `< 5.0`) and fix the code to match it.

- [ ] **Step 5: Lint and commit**

```bash
uvx ruff@0.16.7 check --fix . && uvx ruff@0.16.7 format . && uvx ruff@0.16.7 check --fix . && uvx ruff@0.16.7 format . && uvx ruff@0.16.7 check . && uvx ruff@0.16.7 format --check .
git add houdini/python/fxmotion/client.py houdini/python/fxmotion/paths.py tests/test_client.py tests/test_paths.py tests/test_imports.py
git commit -m "feat(fxmotion): add the server client and the path and timing helpers

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: move the timeline into fxmotion and put Regenerate on the clip format

**Files:**
- Move: `houdini/python/kimodo_timeline/{__init__,model,bridge,widget,regen}.py` -> `houdini/python/fxmotion/timeline/`; `.../kimodo_timeline/qt.py` -> `houdini/python/fxmotion/qt.py`; `.../kimodo_timeline/poller.py` -> `houdini/python/fxmotion/poller.py`
- Move: `houdini/python_panels/kimodo_timeline.pypanel` -> `houdini/python_panels/fxmotion_timeline.pypanel`
- Modify: the moved files as listed below; `tests/test_timeline_model.py`, `tests/test_regen_splice.py`, `tests/test_panel_conventions.py`, `tests/test_houdini_live.py`, `tests/test_imports.py`

**Interfaces:**
- Consumes: `fxmotion.client` (`job`, `cancel`, `submit`, `download`, `ServerError`), `fxmotion.clipformat` (`load`, `save`, `validate`, `meta`, `frame_count`).
- Produces: `fxmotion.timeline.create_widget()`; `fxmotion.timeline.bridge.TIMELINE_TYPES = ("vb::kimodo_motion::2.0",)`; `Timeline.request_segments(fps) -> [{"prompt", "duration_s"}]`; `fxmotion.poller.JobWatcher(node, url, job_id, label, on_done)` with `on_done(data: dict, suffix: str)`; regen pure functions `per_sample_keys(clip) -> list[str]`, `splice(old, new, cut, n, resume_at=None) -> dict`, `continue_payload(clip, cut, n) -> dict` (keys `native_local_rot_mats`, `native_root_positions`), `tail_pin(clip, cut_end, n, at_index) -> list[dict]`, `seam_jump`, `mean_step`, `cut_sample`, `samples_for` (unchanged signatures); `regen.regenerate(node, seg_index, to_end=False) -> str`; `regen.Precondition`. Task 9 provides `fxmotion.nodes.common.set_clip` and `fxmotion.nodes.timeline_parms.rebuild_segments`, imported inside functions here.

- [ ] **Step 1: Move the files**

```bash
mkdir -p houdini/python/fxmotion/timeline
for f in __init__ model bridge widget regen; do git mv houdini/python/kimodo_timeline/$f.py houdini/python/fxmotion/timeline/$f.py; done
git mv houdini/python/kimodo_timeline/qt.py houdini/python/fxmotion/qt.py
git mv houdini/python/kimodo_timeline/poller.py houdini/python/fxmotion/poller.py
git mv houdini/python_panels/kimodo_timeline.pypanel houdini/python_panels/fxmotion_timeline.pypanel
rmdir houdini/python/kimodo_timeline 2>/dev/null; ls houdini/python
```

Expected: `fxmotion` only (plus `__pycache__` if present; delete it).

- [ ] **Step 2: Update the tests first (they now fail)**

In `tests/test_timeline_model.py`: replace `from kimodo_timeline.model import` with `from fxmotion.timeline.model import`, the docstring's `kimodo_timeline.model` with `fxmotion.timeline.model`, and the two expected dicts with `{"prompt": "walk", "duration_s": 2.4}` and `{"prompt": "box", "duration_s": 1.6}`.

In `tests/test_panel_conventions.py`:
- replace lines 21-22 with

```python
FX = HERE.parent / "houdini" / "python" / "fxmotion"
PKG = FX / "timeline"
NODES = FX / "nodes"
```

- in `_modules()`, replace `for path in sorted(PKG.glob("*.py")):` with `for path in sorted([*PKG.glob("*.py"), FX / "poller.py", FX / "qt.py"]):`
- in `test_no_interruptable_operation_in_the_panel` and `test_nothing_sleeps`, replace the `_code(_read(HDA_SRC))` assertion with a loop over the node modules:

```python
    for path in sorted(NODES.glob("*.py")):
        assert "InterruptableOperation" not in _code(_read(path)), (
            "%s must not block on an operation" % path.name
        )
```

(and the same shape with `"time.sleep"` / `"must not sleep"` in `test_nothing_sleeps`)
- in the ASCII test (line ~300) replace `sorted(PKG.glob("*.py"))` with `sorted(FX.rglob("*.py"))`
- in `test_poller_never_touches_hou_ui_at_import`, replace `PKG / "poller.py"` with `FX / "poller.py"`
- in `test_a_precondition_is_a_warning_not_an_error`, replace the two lines from `hda = _read(HDA_SRC)` to the `for src, fn in (` tuple's second entry with:

```python
    for src, fn in (
        (_code(_read(PKG / "widget.py")), "_regen"),
        (_code(_read(NODES / "timeline_parms.py")), "run_regenerate"),
    ):
```

In `tests/test_houdini_live.py`: replace every `kimodo_timeline` with `fxmotion.timeline` in imports, except `from kimodo_timeline.qt import` which becomes `from fxmotion.qt import`.

Replace `tests/test_regen_splice.py` from `from kimodo_timeline.regen import (` to the end of the file with:

```python
from cliptools import tiny  # noqa: E402

from fxmotion import clipformat  # noqa: E402
from fxmotion.timeline.regen import (  # noqa: E402
    continue_payload,
    cut_sample,
    per_sample_keys,
    samples_for,
    splice,
)

# scene rates a shot gets cut at, against the 30 the SOMA models generate at,
# plus a couple of mismatched pairs to keep the rounding honest
FPS_PAIRS = ((24, 30), (25, 30), (30, 30), (24, 20), (60, 30))


def test_cut_lines_up_with_what_the_server_generates():
    """The slot a segment occupies in the clip has to be exactly as long as
    the segment the server generates for it (int(duration * fps) per
    segment, mirrored by samples_for), or every single-segment regen drops a
    sample."""
    for scene, source in FPS_PAIRS:
        for f in range(4, 120):
            frames = [f, f + 1, f + 7, f + 13]
            for i in range(1, len(frames)):
                cut = cut_sample(frames, i, scene, source)
                resume = cut_sample(frames, i + 1, scene, source)
                nf = samples_for(frames[i], scene, source)
                assert resume - cut == nf


def _clip(length, value=None):
    """A valid clip whose per-sample values say which sample they came from,
    so a misplaced join shows up as a wrong number rather than a right
    shape."""
    col = np.arange(length, dtype=float) if value is None else np.full(length, value)
    pos = np.zeros((length, 2, 3))
    pos[:, :, 0] = col[:, None]
    return tiny(
        length,
        world_pos=pos,
        native={"root_positions": np.repeat(col[:, None], 3, axis=1)},
    )


def test_per_sample_keys():
    assert set(per_sample_keys(_clip(4))) == {
        "world_pos",
        "world_rot",
        "native_root_positions",
    }


def test_single_segment_splice_keeps_the_clip_length_and_stays_valid():
    scene, source, n = 24, 30, 5
    frames = [31, 47, 23]
    have = cut_sample(frames, len(frames), scene, source)
    old = _clip(have)
    cut = cut_sample(frames, 1, scene, source)
    resume = cut_sample(frames, 2, scene, source)
    nf = samples_for(frames[1], scene, source)
    new = _clip(n + nf, value=-1.0)  # the blended transition, then the segment

    merged = splice(old, new, cut, n, resume_at=resume)
    clipformat.validate(merged)
    assert clipformat.frame_count(merged) == have
    x = merged["world_pos"][:, 0, 0]
    assert np.array_equal(x[: cut - n], np.arange(cut - n))
    assert np.array_equal(x[cut + nf :], np.arange(resume, have))
    assert (x[cut - n : cut + nf] == -1.0).all()


def test_run_to_the_end_splice_drops_the_old_tail():
    scene, source, n = 24, 30, 5
    frames = [31, 47, 23]
    have = cut_sample(frames, len(frames), scene, source)
    cut = cut_sample(frames, 1, scene, source)
    merged = splice(_clip(have), _clip(n + have - cut, -1.0), cut, n)
    assert clipformat.frame_count(merged) == have


def test_continue_payload_sends_the_native_tail():
    clip = tiny(
        40,
        native={
            "root_positions": np.arange(120, dtype=float).reshape(40, 3),
            "local_rot_mats": np.tile(np.eye(3), (40, 2, 1, 1)),
        },
    )
    body = continue_payload(clip, cut=30, n=5)
    assert set(body) == {"native_local_rot_mats", "native_root_positions"}
    assert np.array_equal(body["native_root_positions"], clip["native_root_positions"][25:30].tolist())
```

and keep the `if __name__ == "__main__":` runner block at the bottom unchanged.

Append `"fxmotion.timeline.model"` and `"fxmotion.timeline.regen"` to `PURE` in `tests/test_imports.py`.

Run: `python -m pytest -q`
Expected: FAIL (regen functions `per_sample_keys` missing, `request_segments` key mismatch, `timeline_parms.py` not found).

- [ ] **Step 3: Edit the moved modules**

`fxmotion/timeline/__init__.py`: replace its docstring with `"""The Motion Timeline panel for fxmotion generator nodes."""` and the entry-point docstring with `"""Entry point used by houdini/python_panels/fxmotion_timeline.pypanel."""`.

`fxmotion/timeline/widget.py`: replace `from .qt import QtCore, QtGui, QtWidgets, event_pos, run_exec` with `from ..qt import QtCore, QtGui, QtWidgets, event_pos, run_exec`.

`fxmotion/timeline/model.py`: in `request_segments`, replace `"duration": s.frames / float(fps)` with `"duration_s": s.frames / float(fps)`.

`fxmotion/timeline/bridge.py`: replace `TYPE_PREFIX = "vb::kimodo_motion"` with

```python
# Node types the panel edits. 1.1 is left out: it can no longer Generate.
TIMELINE_TYPES = ("vb::kimodo_motion::2.0",)


def _is_timeline_node(n) -> bool:
    return n is not None and n.type().name() in TIMELINE_TYPES
```

then in `find_node` use `if _is_timeline_node(n):`, in `all_nodes` use `if name in TIMELINE_TYPES:`, in `node_at` use `return n if _is_timeline_node(n) else None`, change `save`'s default label to `"Motion timeline edit"`, and replace `node.type().hdaModule().rebuild_segments(node)` with

```python
        from ..nodes import timeline_parms

        timeline_parms.rebuild_segments(node)
```

`fxmotion/poller.py`: replace `import requests` with `from . import client` (keep `import hou`), replace `from .qt import QtWidgets` (already correct relative to the new location, leave it), change the docstring's first line to `"""Watch a model-server job from Houdini's event loop, without blocking anything.`, replace `self.dlg = JobDialog("Kimodo")` with `self.dlg = JobDialog(self.node.type().description())`, replace the `hou.OperationFailed` message with `"fxmotion polls the job from Houdini's event loop, which needs a UI session."`, replace `"Kimodo: %s" % msg` in `fail` with `"%s: %s" % (self.node.type().description(), msg)`, and in `_poll_once` replace the cancel `requests.post(...)` call with `client.cancel(self.url, self.job_id)` and the block from `try:` through the `except requests.RequestException as e:` handler with:

```python
        try:
            data = client.job(self.url, self.job_id)
            self._fails = 0
        except client.ServerError as e:
            if e.status == 404:
                self.fail("Job lost (server restarted?)")
                return
            self._fails += 1
            self.node.parm("status").set(
                "Poll error (%d/%d): %s" % (self._fails, MAX_FAILS, e)
            )
            if self._fails >= MAX_FAILS:
                self.fail("Lost contact with the server while polling: %s" % e)
            return
```

`houdini/python_panels/fxmotion_timeline.pypanel`: replace the whole file with

```xml
<?xml version="1.0" encoding="UTF-8"?>
<pythonPanelDocument>
  <!-- Motion Timeline: prompt segments and pose tracks for fxmotion generator
       nodes. The widget code lives in houdini/python/fxmotion/timeline (on
       PYTHONPATH via the fxhoudinimotion package). -->
  <interface name="fxmotion_timeline" label="Motion Timeline" icon="NVIDIA_badge.svg" showNetworkNavigationBar="false" help_url="">
    <script><![CDATA[
def onCreateInterface():
    from fxmotion import timeline
    return timeline.create_widget()
]]></script>
    <includeInToolbarMenu menu_position="0" create_separator="false"/>
  </interface>
</pythonPanelDocument>
```

- [ ] **Step 4: Rewrite regen on the clip format**

In `fxmotion/timeline/regen.py`, keep the module docstring, `Precondition`, `cut_sample`, `samples_for`, and replace everything else with:

```python
from __future__ import annotations

import numpy as np

from .. import clipformat

# Per-sample keys every clip has; native_* keys are per-sample when their
# first axis is the clip's length.
PER_SAMPLE = ("world_pos", "world_rot", "contacts")


def per_sample_keys(clip) -> list:
    frames = clipformat.frame_count(clip)
    return [
        k
        for k, v in clip.items()
        if (k in PER_SAMPLE or k.startswith("native_"))
        and getattr(v, "ndim", 0) >= 1
        and v.shape[0] == frames
    ]
```

then `Precondition`, `cut_sample` unchanged, then:

```python
def continue_payload(clip, cut: int, n: int) -> dict:
    """The `n` samples before `cut`, in Kimodo's model space, as the
    request's `continue_from` block. Local rotations rather than global:
    they are what the motion representation is built from, and the server's
    77 -> model-skeleton slice is the exact inverse of the output
    conversion, so the round trip is lossless."""
    if cut < n:
        raise ValueError("need %d samples before the cut, cut is at %d" % (n, cut))
    head = slice(cut - n, cut)
    return {
        "native_local_rot_mats": clip["native_local_rot_mats"][head].tolist(),
        "native_root_positions": clip["native_root_positions"][head].tolist(),
    }


def splice(old, new, cut: int, n: int, resume_at=None) -> dict:
    """Keep `old` up to the transition, then the returned clip whole.

    `new`'s first `n` samples are the blended seam the model produced for
    those frames, so they replace old[cut - n:cut] rather than being
    dropped. `resume_at` continues with the rest of the original clip
    afterwards, for a single-segment replacement. Anything not per-sample
    (skeleton, rest pose, source) rides along from `old`."""
    out = dict(old)
    for k in per_sample_keys(old):
        if k in new:
            parts = [old[k][: cut - n], new[k]]
            if resume_at is not None:
                parts.append(old[k][resume_at:])
            out[k] = np.concatenate(parts, axis=0)
    out.pop("segments", None)  # its ranges no longer describe the clip
    return out


def seam_jump(merged, at: int) -> float:
    """Max joint movement across the join at sample `at`."""
    pj = merged["world_pos"]
    return float(np.linalg.norm(pj[at] - pj[at - 1], axis=-1).max())


def mean_step(clip) -> float:
    """The clip's own mean per-sample joint movement, the scale a seam is
    judged against."""
    d = np.linalg.norm(np.diff(clip["world_pos"], axis=0), axis=-1).max(axis=1)
    return float(d.mean())


def tail_pin(clip, cut_end: int, n: int, at_index: int) -> list:
    """Constraints holding the new segment's last `n` frames to the `n`
    samples before `cut_end`, in model space (Kimodo native constraint
    dicts), so whatever follows in the existing clip still joins on.
    Indices are segment-relative: the model crops user constraints with
    current_frame = 0 for the first requested segment."""
    head = slice(cut_end - n, cut_end)
    common = {
        "frame_indices": list(range(at_index, at_index + n)),
        "global_joints_positions": clip["native_posed_joints"][head].tolist(),
        "global_joints_rots": clip["native_global_rot_mats"][head].tolist(),
        "smooth_root_2d": clip["native_smooth_root_pos"][head][:, [0, 2]].tolist(),
    }
    return [
        dict(common, type="fullbody-global"),
        dict(
            common,
            type="ee-global",
            joint_names=["LeftHand", "RightHand", "LeftFoot", "RightFoot"],
        ),
    ]
```

then `samples_for` unchanged (drop `load_npz_bytes` and `CLIP_KEYS`, no longer used), then the Houdini half:

```python
###### Houdini side
# Below the pure functions so the module still imports without hou.


def regenerate(node, seg_index: int, to_end: bool = False):
    """Re-roll segment `seg_index` (0-based), in place.

    By default only that segment: its head is joined with continue_from and
    its tail held to the frames the next segment was generated against, so
    the clip keeps its length and everything either side is untouched.
    Measured on a 601-sample clip, that leaves the two joins at 0.80x and
    0.39x of the clip's own per-sample motion, against 31.83x with no pin.
    `to_end` re-rolls this segment and every one after it instead.

    Returns as soon as the job is queued; a JobWatcher finishes the merge
    when the clip arrives.
    """
    from pathlib import Path

    import hou

    from .. import client
    from ..nodes import common
    from ..poller import JobWatcher
    from .model import Timeline

    tl = Timeline.from_json(node.parm("timeline_json").eval())
    if not tl.segments:
        raise Precondition("This node has no timeline to regenerate from.")
    if not 0 <= seg_index < len(tl.segments):
        raise Precondition("Segment %d is outside the timeline." % (seg_index + 1))
    if seg_index == 0:
        raise Precondition(
            "Segment 1 has no earlier motion to continue from; use Generate."
        )
    src = node.parm("clip_path").eval()
    if not src or not Path(src).exists():
        raise Precondition("No generated clip on this node yet. Press Generate first.")
    old = clipformat.load(src)
    if "native_local_rot_mats" not in old:
        raise Precondition(
            "This clip carries no Kimodo motion history; Regenerate needs a "
            "clip from the Kimodo server. Press Generate first."
        )

    n = max(1, int(tl.transition_frames))
    scene_fps = float(hou.fps())
    source_fps = float(old["fps"])
    frames = [s.frames for s in tl.segments]
    cut = cut_sample(frames, seg_index, scene_fps, source_fps)
    have = clipformat.frame_count(old)
    # A bounds check is not enough: edited segment lengths still give an
    # in-range cut, just the wrong one. Compare the timeline to the file.
    expect = cut_sample(frames, len(frames), scene_fps, source_fps)
    if abs(expect - have) > 2 * n:
        raise Precondition(
            "The clip on disk is %d samples but this timeline describes %d. "
            "The segment lengths changed since it was generated, so the cut "
            "would land in the wrong place. Press Generate first." % (have, expect)
        )
    if cut - n < 1 or cut > have:
        raise Precondition(
            "Segment %d falls outside the clip on disk; press Generate." % (seg_index + 1)
        )

    single = not to_end and seg_index + 1 < len(tl.segments)
    send = tl.segments[seg_index : seg_index + 1] if single else tl.segments[seg_index:]
    source = clipformat.meta(old, "source") or {}
    options = {"transition_frames": n, "canon": source.get("canon")}
    resume_at = None
    if single:
        resume_at = cut_sample(frames, seg_index + 1, scene_fps, source_fps)
        nf = samples_for(frames[seg_index], scene_fps, source_fps)
        options["native_constraints"] = tail_pin(old, resume_at, n, nf - n)
    body = {
        "segments": Timeline(send).request_segments(scene_fps),
        "model": node.parm("model").evalAsString(),
        "force": bool(node.parm("force").eval()),
        "continue_from": continue_payload(old, cut, n),
        "options": options,
    }
    url = common.url(node)
    try:
        job = client.submit(url, body)
    except client.Rejected as e:
        raise Precondition("The server refused the regeneration: %s" % e) from e

    what = "segment %d" % (seg_index + 1) if single else "from segment %d" % (seg_index + 1)

    def _merge(data, suffix):
        """Runs on the main thread from the event-loop callback when done."""
        try:
            got = client.download(url, job, node.parm("download_dir").eval())
            new = clipformat.load(got)
        except (client.ServerError, clipformat.ClipError) as e:
            common.fail(node, "Regenerated clip unusable: %s" % e)
            return
        merged = splice(old, new, cut, n, resume_at=resume_at)
        head_at = cut - n
        normal = mean_step(old)
        jump = seam_jump(merged, head_at)
        joins, worst = "%.2f cm" % (jump * 100), jump
        if resume_at is not None:  # a single segment has two joins
            tail = seam_jump(merged, head_at + clipformat.frame_count(new))
            joins, worst = "%.2f / %.2f cm" % (jump * 100, tail * 100), max(jump, tail)
        out = Path(src).parent / ("%s_joined.npz" % job)
        clipformat.save(out, merged)
        common.set_clip(
            node,
            out.as_posix(),
            "Regenerated %s; seam %s vs %.2f cm/sample (%.2fx)"
            % (what, joins, normal * 100, worst / normal),
        )

    node.parm("job_id").set(job)
    node.parm("progress").set(0.0)
    node.parm("status").set("Queued (%s...)" % job[:8])
    JobWatcher(node, url, job, "Regenerating " + what, _merge).start()
    return "Regenerating %s; watch the node's Status." % what
```

- [ ] **Step 5: Run the offline tests**

Run: `python -m pytest -q --deselect tests/test_panel_conventions.py::test_a_precondition_is_a_warning_not_an_error --deselect tests/test_panel_conventions.py::test_no_interruptable_operation_in_the_panel --deselect tests/test_panel_conventions.py::test_nothing_sleeps`
Expected: all pass. The three deselected tests need `fxmotion/nodes/`, which Task 9 creates; they must pass at the end of Task 9.

- [ ] **Step 6: Lint and commit**

```bash
uvx ruff@0.16.7 check --fix . && uvx ruff@0.16.7 format . && uvx ruff@0.16.7 check --fix . && uvx ruff@0.16.7 format . && uvx ruff@0.16.7 check . && uvx ruff@0.16.7 format --check .
git add -A houdini/python houdini/python_panels tests
git commit -m "refactor(timeline): move the panel into fxmotion and regenerate on clips

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 9: node logic (generator plumbing and the Kimodo node)

**Files:**
- Create: `houdini/python/fxmotion/nodes/__init__.py`, `nodes/common.py`, `nodes/kimodo.py`, `nodes/timeline_parms.py`
- Test: `tests/test_panel_conventions.py` (the three deselected tests), `tests/test_nodes_source.py`

**Interfaces:**
- Consumes: `fxmotion.client`, `fxmotion.clip`, `fxmotion.clipformat`, `fxmotion.paths`, `fxmotion.skeletons`, `fxmotion.poller.JobWatcher`, `fxmotion.timeline.regen`.
- Produces: `common.url(node)`, `common.label(node)`, `common.say(node, msg, severity=None)`, `common.fail(node, msg)`, `common.start_job(node, payload, what="Running")`, `common.set_clip(node, path, status)`, `common.cancel(node)`, `common.test_connection(node)`, `common.cook_animated(sop)`, `common.cook_rest(sop, skeleton)`, `common.cook_file(sop, skeleton, which)`, `common.load_skeleton_file(sop, skeleton, which)`; `timeline_parms.read_timeline(node)`, `write_timeline(node, tl)`, `segments_from_parms(node)`, `rebuild_segments(node)`, `refresh_starts(node, segs=None)`, `sync_from_parms(node)`, `split_segment(node, index)`, `run_regenerate(node, index, to_end=False)`, `on_created(node)`; `kimodo.SKELETON = "soma77"`, `kimodo.generate(node)`, `kimodo.build_payload(node, clip_fps) -> dict`, `kimodo.make_pose_rig(node)`, `kimodo.open_timeline(node)`, and re-exports `kimodo.cancel`, `kimodo.test_connection`, `kimodo.sync_segments`, `kimodo.split_segment`, `kimodo.regenerate`, `kimodo.refresh_starts`, `kimodo.on_created` for the HDA callbacks. Parm names the HDA (Task 10) must define: `server_url, download_dir, model, force, seed, prompt, duration_frames, start_frame, retime, clip_path, clip_info, status, progress, job_id, last_error, timeline_json, has_timeline, frame_ref, scene_fps, segments (multiparm: seg_prompt#, seg_frames#, seg_from#, seg_to#), path_waypoints, constraints_json, constraints_file, pose_keyframes, pose_type, ee_left_hand, ee_right_hand, ee_left_foot, ee_right_foot`.

- [ ] **Step 1: Write the failing source test**

The node modules need hou, so their behaviour is covered by the live test in Task 10. This offline test pins what is checkable without Houdini: every callback name the HDA calls exists.

`tests/test_nodes_source.py`:

```python
"""The HDA callbacks call names in fxmotion.nodes.kimodo and .common; a
rename that forgets the asset breaks every button silently. Checked from
source, without hou."""

import ast
from pathlib import Path

NODES = Path(__file__).resolve().parents[1] / "houdini" / "python" / "fxmotion" / "nodes"

CALLBACKS = {
    "kimodo.py": {
        "generate",
        "cancel",
        "test_connection",
        "make_pose_rig",
        "open_timeline",
        "sync_segments",
        "split_segment",
        "regenerate",
        "refresh_starts",
        "on_created",
    },
    "common.py": {"cook_animated", "cook_rest", "cook_file", "load_skeleton_file"},
}


def _names(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            out.add(node.name)
        elif isinstance(node, ast.ImportFrom):
            out.update(a.asname or a.name for a in node.names)
        elif isinstance(node, ast.Assign):
            out.update(t.id for t in node.targets if isinstance(t, ast.Name))
    return out


def test_every_hda_callback_exists():
    for module, wanted in CALLBACKS.items():
        missing = wanted - _names(NODES / module)
        assert not missing, "%s lacks %s" % (module, sorted(missing))
```

Run: `python -m pytest tests/test_nodes_source.py -q`
Expected: FAIL with `FileNotFoundError` on `kimodo.py`.

- [ ] **Step 2: Write `nodes/__init__.py` and `nodes/common.py`**

`houdini/python/fxmotion/nodes/__init__.py`:

```python
"""What the generator HDAs call. Each HDA callback and each SOP cook inside
an asset is one line into this package, so the logic is here, in version
control, not in script strings inside the asset. Needs hou."""
```

`houdini/python/fxmotion/nodes/common.py`:

```python
"""Generator-node plumbing shared by every model: jobs, errors and cooks."""

from __future__ import annotations

import hou

from .. import client, clip, clipformat, skeletons


def url(node) -> str:
    return node.parm("server_url").eval().rstrip("/")


def label(node) -> str:
    return node.type().description()


def say(node, msg, severity=None) -> None:
    """Status parm (shown under the node and in the panel) plus the status bar."""
    node.parm("status").set(str(msg))
    if hou.isUIAvailable():
        hou.ui.setStatusMessage(
            "%s: %s" % (label(node), msg),
            severity=severity or hou.severityType.Message,
        )


def fail(node, msg) -> None:
    """A real failure: last_error turns the node red (see raise_last_error)."""
    node.parm("last_error").set(str(msg))
    node.parm("job_id").set("")
    say(node, "Error: %s" % msg, hou.severityType.Error)


def start_job(node, payload, what="Running") -> None:
    """Submit `payload` and poll it from the event loop; the node recooks
    when the clip lands. Houdini stays interactive throughout."""
    from ..poller import JobWatcher

    job_id = client.submit(url(node), payload)
    node.parm("job_id").set(job_id)
    node.parm("progress").set(0.0)
    node.parm("status").set("Queued (%s...)" % job_id[:8])

    def _done(data, suffix):
        try:
            path = client.download(url(node), job_id, node.parm("download_dir").eval())
            cached = " (cached)" if data.get("cached") else ""
            set_clip(node, path, "Done%s%s" % (suffix, cached))
        except (client.ServerError, clipformat.ClipError, OSError) as e:
            fail(node, "Clip download failed: %s" % e)

    JobWatcher(node, url(node), job_id, what, _done).start()


def set_clip(node, path, status) -> None:
    """Point the node at a clip on disk and recook."""
    c = clip.load(path)
    frames, fps = clipformat.frame_count(c), float(c["fps"])
    secs = frames / fps
    node.parm("clip_info").set(
        "%.2f s = %d frames @ %g fps (%d samples @ %g fps)"
        % (secs, round(secs * hou.fps()), hou.fps(), frames, fps)
    )
    node.parm("clip_path").set(path)
    node.parm("job_id").set("")
    node.parm("progress").set(1.0)
    node.parm("last_error").set("")
    say(node, status, hou.severityType.ImportantMessage)
    node.cook(force=True)


def cancel(node) -> None:
    job_id = node.parm("job_id").eval()
    if not job_id:
        say(node, "No active job to cancel.", hou.severityType.Warning)
        return
    try:
        client.cancel(url(node), job_id)
    except client.ServerError as e:
        say(node, "Cancel failed: %s" % e, hou.severityType.Error)
        return
    node.parm("job_id").set("")
    say(node, "Cancelled")


def test_connection(node) -> None:
    try:
        h = client.health(url(node))
    except client.ServerError as e:
        say(node, "Server unreachable: %s" % e, hou.severityType.Error)
        return
    mock = " (mock mode, no inference)" if h.get("mock_mode") else ""
    say(
        node,
        "Server OK: %s, %s at %g fps%s" % (h["backend"], h["skeleton"], h["fps"], mock),
        hou.severityType.ImportantMessage,
    )


###### Cooks: called by the Python SOPs inside the asset


def raise_last_error(hda) -> None:
    """A failed Generate leaves its message in last_error; every output
    raises it, so the node goes red with the message in its info."""
    err = hda.parm("last_error").eval().strip()
    if err:
        raise hou.NodeError(err)


def cook_animated(sop) -> None:
    hda = sop.parent()
    raise_last_error(hda)
    path = hda.parm("clip_path").eval()
    if not path:
        return  # nothing generated yet: empty output, wait for Generate
    c = clip.load(path)
    index = clip.sample_index(
        hda.parm("frame_ref").eval(),
        hda.parm("start_frame").eval(),
        hou.fps(),
        float(c["fps"]),
        bool(hda.parm("retime").eval()),
        clipformat.frame_count(c),
    )
    clip.build(sop.geometry(), c, index)


def cook_rest(sop, skeleton) -> None:
    raise_last_error(sop.parent())
    clip.build_rest(sop.geometry(), skeletons.get(skeleton))


def load_skeleton_file(sop, skeleton, which) -> None:
    """The skeleton's `skin` or `capture_pose` file, into this SOP."""
    path = getattr(skeletons.get(skeleton), which)
    if path is None:
        raise hou.NodeError("skeleton %s has no %s geometry" % (skeleton, which))
    sop.geometry().loadFromFile(str(path))


def cook_file(sop, skeleton, which) -> None:
    raise_last_error(sop.parent())
    load_skeleton_file(sop, skeleton, which)
```

- [ ] **Step 3: Write `nodes/timeline_parms.py`**

```python
"""Segments multiparm <-> timeline_json.

The JSON stays canonical because it also holds the pose-key tracks, which
have no sensible multiparm form; the multiparm is a real editor over the
segments that writes back. Ported from the 1.1 HDA's PythonModule.
"""

from __future__ import annotations

import json

import hou

from . import common


def read_timeline(node) -> dict:
    raw = node.parm("timeline_json").eval().strip()
    try:
        return json.loads(raw) if raw else {}
    except ValueError:
        return {}


def write_timeline(node, tl) -> bool:
    """Write the JSON and the parms it mirrors. True if anything changed."""
    segs = tl.get("segments") or []
    tl["version"] = 1
    tl.setdefault("transition_frames", 5)
    tl.setdefault("tracks", {})
    out = json.dumps(tl, indent=1)
    if out == node.parm("timeline_json").eval():
        return False
    node.parm("timeline_json").set(out)
    node.parm("has_timeline").set(1 if segs else 0)
    if segs:
        node.parm("duration_frames").set(sum(int(s["frames"]) for s in segs))
    return True


def segments_from_parms(node) -> list:
    return [
        {
            "prompt": node.parm("seg_prompt%d" % i).eval(),
            "frames": max(1, int(node.parm("seg_frames%d" % i).eval())),
        }
        for i in range(1, int(node.parm("segments").eval()) + 1)
    ]


def refresh_starts(node, segs=None) -> None:
    """Fill each instance's read-only first and last scene frame."""
    start = node.parm("start_frame")
    if start is None:  # never take the node down over a display field
        return
    if segs is None:
        segs = read_timeline(node).get("segments") or []
    f = int(start.eval())
    for i, sg in enumerate(segs, start=1):
        a, b = node.parm("seg_from%d" % i), node.parm("seg_to%d" % i)
        if a is None or b is None:
            break
        last = f + int(sg["frames"]) - 1
        if a.eval() != f:
            a.set(f)
        if b.eval() != last:
            b.set(last)
        f = last + 1


def rebuild_segments(node) -> None:
    """Push timeline_json's segments into the multiparm, without churning
    parms that already match (each set is an undo entry and a recook)."""
    segs = read_timeline(node).get("segments") or []
    p = node.parm("segments")
    if p.eval() != len(segs):
        p.set(len(segs))
    for i, s in enumerate(segs, start=1):
        pp, pf = node.parm("seg_prompt%d" % i), node.parm("seg_frames%d" % i)
        if pp.eval() != s["prompt"]:
            pp.set(s["prompt"])
        if pf.eval() != int(s["frames"]):
            pf.set(int(s["frames"]))
    refresh_starts(node, segs)


def sync_from_parms(node) -> bool:
    """Multiparm edited by hand: fold it back into the JSON."""
    tl = read_timeline(node)
    new = segments_from_parms(node)
    if not new and tl.get("segments"):
        # emptied: fall back to a single prompt, keeping the first one
        node.parm("prompt").set(tl["segments"][0]["prompt"])
    tl["segments"] = new
    changed = write_timeline(node, tl)
    refresh_starts(node, new)
    return changed


def split_segment(node, index) -> None:
    """Cut segment `index` (0-based) in two at the playhead."""
    tl = read_timeline(node)
    segs = tl.get("segments") or []
    if not 0 <= index < len(segs):
        return
    start = int(node.parm("start_frame").eval()) + sum(int(s["frames"]) for s in segs[:index])
    left = int(round(hou.frame())) - start
    if 0 < left < int(segs[index]["frames"]):
        segs.insert(index + 1, {"prompt": segs[index]["prompt"], "frames": int(segs[index]["frames"]) - left})
        segs[index]["frames"] = left
        tl["segments"] = segs
        with hou.undos.group("%s: split segment" % common.label(node)):
            write_timeline(node, tl)
            rebuild_segments(node)
    elif hou.isUIAvailable():
        hou.ui.setStatusMessage(
            "%s: put the playhead inside this segment to split it." % common.label(node),
            severity=hou.severityType.Warning,
        )


def run_regenerate(node, index, to_end=False) -> None:
    """Shared body for the two Regenerate buttons."""
    from ..timeline import regen

    try:
        msg = regen.regenerate(node, index, to_end=to_end)
    except regen.Precondition as e:
        # A wrong turn, not a broken cook: last_error is left alone, since
        # the cook turns it into a hou.NodeError and reddens the node.
        node.parm("status").set(str(e))
        if hou.isUIAvailable():
            hou.ui.setStatusMessage(
                "%s: %s" % (common.label(node), e), severity=hou.severityType.Warning
            )
    except Exception as e:
        common.fail(node, e)
    else:
        node.parm("last_error").set("")
        if hou.isUIAvailable():
            hou.ui.setStatusMessage(
                "%s: %s" % (common.label(node), msg),
                severity=hou.severityType.ImportantMessage,
            )


def on_created(node) -> None:
    """Node shape, and the first segment from the defaults. Keyed on the
    timeline being empty, not on the instance count: the multiparm defaults
    to 1, so a count test never fires."""
    node.setUserData("nodeshape", "bulge")
    if not node.parm("timeline_json").eval().strip():
        if node.parm("segments").eval() < 1:
            node.parm("segments").set(1)
        if not node.parm("seg_prompt1").eval():
            node.parm("seg_prompt1").set(node.parm("prompt").eval())
            node.parm("seg_frames1").set(int(node.parm("duration_frames").eval()))
        sync_from_parms(node)
```

- [ ] **Step 4: Write `nodes/kimodo.py`**

```python
"""The Kimodo Motion node: its parms and inputs as a request."""

from __future__ import annotations

import json

import hou
import numpy as np

from .. import client, paths, skeletons
from . import common, timeline_parms
from .common import cancel, test_connection  # noqa: F401  (HDA callbacks)
from .timeline_parms import on_created, refresh_starts, split_segment  # noqa: F401

SKELETON = "soma77"
EFFECTOR_PARMS = (
    ("ee_left_hand", "LeftHand"),
    ("ee_right_hand", "RightHand"),
    ("ee_left_foot", "LeftFoot"),
    ("ee_right_foot", "RightFoot"),
)


def generate(node) -> None:
    node.parm("last_error").set("")
    try:
        info = client.health(common.url(node))
        common.start_job(node, build_payload(node, float(info["fps"])))
    except client.Rejected as e:
        common.fail(node, "The server refused the request: %s" % e)
    except Exception as e:
        common.fail(node, e)
    else:
        if hou.isUIAvailable():
            hou.ui.setStatusMessage(
                "%s: generation started, watch the node's Status." % common.label(node),
                severity=hou.severityType.ImportantMessage,
            )


def build_payload(node, clip_fps) -> dict:
    scene_fps = hou.fps()
    start = node.parm("start_frame").eval()
    retime = bool(node.parm("retime").eval())

    def secs(frame):
        return paths.scene_seconds(frame, start, scene_fps, clip_fps, retime)

    tl = timeline_parms.read_timeline(node)
    segs = tl.get("segments") or []
    if segs:
        segments = [
            {"prompt": s["prompt"].strip(), "duration_s": int(s["frames"]) / scene_fps}
            for s in segs
        ]
    else:
        segments = [
            {
                "prompt": node.parm("prompt").eval().strip(),
                "duration_s": node.parm("duration_frames").eval() / scene_fps,
            }
        ]
    payload = {
        "segments": segments,
        "model": node.parm("model").evalAsString(),
        "force": bool(node.parm("force").eval()),
        "options": {"transition_frames": int(tl.get("transition_frames", 5))},
    }
    seed = int(node.parm("seed").eval())
    if seed >= 0:
        payload["seed"] = seed
    native = _native_constraints(node)
    if native:
        payload["options"]["native_constraints"] = native
    root = _root_path(node, secs)
    if root:
        payload["root_path"] = root
    keys = _keyframes(node, tl, secs)
    if keys:
        payload["keyframes"] = keys
    return payload


def _native_constraints(node):
    """Constraints JSON (inline wins) or file: raw Kimodo dicts, sent in
    Kimodo's model space as authored."""
    raw = node.parm("constraints_json").eval().strip()
    if not raw:
        path = node.parm("constraints_file").eval().strip()
        if path:
            with open(path, encoding="utf-8") as fh:
                raw = fh.read()
    if not raw:
        return None
    data = json.loads(raw)
    if not isinstance(data, list):
        raise ValueError("Constraints must be a JSON list of constraint dicts.")
    return data


def _root_path(node, secs):
    """Input 0 as root_path points. Points with an int `frame` attribute are
    timed waypoints; otherwise the server spreads them over the clip."""
    ins = node.inputs()
    if not ins or ins[0] is None:
        return None
    geo = node.inputGeometry(0)
    pts = geo.points()
    if not pts:
        return None
    xz = [[p.position()[0], p.position()[2]] for p in pts]
    keep = paths.thin(xz, int(node.parm("path_waypoints").eval()))
    timed = geo.findPointAttrib("frame") is not None
    out = []
    for i in keep:
        p = pts[i].position()
        point = {"pos": [p[0], p[1], p[2]]}
        if timed:
            point["time_s"] = secs(int(pts[i].attribValue("frame")))
        out.append(point)
    return out


def _keyframes(node, tl, secs):
    if tl.get("segments"):
        groups = [
            (None if track == "fullbody" else [track], sorted(int(k) for k in keys))
            for track, keys in (tl.get("tracks") or {}).items()
            if keys
        ]
    else:
        frames = [int(x) for x in node.parm("pose_keyframes").eval().replace(",", " ").split()]
        groups = []
        if frames:
            joints = None
            if node.parm("pose_type").evalAsString() == "End-Effector":
                joints = [j for parm, j in EFFECTOR_PARMS if node.parm(parm).eval()]
                if not joints:
                    raise ValueError(
                        "End-Effector pose constraint: select at least one hand or foot."
                    )
            groups = [(joints, frames)]
    if not groups:
        return []
    ins = node.inputs()
    if len(ins) < 2 or ins[1] is None:
        raise ValueError("Pose keys are set but nothing is wired to input 1 (posed skeleton).")
    out = []
    for joints, frames in groups:
        for f in frames:
            pos, rot = _pose_at(ins[1], f)
            out.append({"time_s": secs(f), "world_pos": pos, "world_rot": rot, "joints": joints})
    return out


def _pose_at(src, frame):
    """World positions and KineFX transforms of every SOMA77 joint on the
    posed skeleton `src`, at a scene frame. The server inverts the
    transform, so nothing is converted here."""
    skel = skeletons.get(SKELETON)
    index = {n: i for i, n in enumerate(skel.joint_names)}
    geo = src.geometryAtFrame(frame)
    if geo.findPointAttrib("name") is None or geo.findPointAttrib("transform") is None:
        raise ValueError(
            "Input 1 must be a SOMA77 skeleton with name and transform point attributes."
        )
    names = geo.pointStringAttribValues("name")
    pos_all = np.asarray(geo.pointFloatAttribValues("P")).reshape(-1, 3)
    rot_all = np.asarray(geo.pointFloatAttribValues("transform")).reshape(-1, 3, 3)
    count = len(skel.joint_names)
    pos, rot = [None] * count, [None] * count
    for k, name in enumerate(names):
        i = index.get(name)
        if i is not None:
            pos[i], rot[i] = pos_all[k].tolist(), rot_all[k].tolist()
    missing = [skel.joint_names[i] for i in range(count) if pos[i] is None]
    if missing:
        raise ValueError(
            "Input 1 is missing %d SOMA77 joints at frame %d, e.g. %s."
            % (len(missing), frame, ", ".join(missing[:5]))
        )
    return pos, rot


def sync_segments(node) -> None:
    timeline_parms.sync_from_parms(node)


def regenerate(node, index, to_end=False) -> None:
    timeline_parms.run_regenerate(node, index, to_end=to_end)


def make_pose_rig(node) -> None:
    """Drop an independent capture-pose rig (+ Rig Pose) and wire it to
    input 1, ready to pose and keyframe."""
    parent = node.parent()
    rig = parent.createNode("python", "pose_rig")
    rig.parm("python").set(
        "from fxmotion.nodes import common\n"
        "common.load_skeleton_file(hou.pwd(), %r, 'capture_pose')\n" % SKELETON
    )
    try:
        tip = parent.createNode("kinefx::rigpose", "pose_keyframes")
        tip.setInput(0, rig)
    except hou.OperationFailed:
        tip = rig
    rig.moveToGoodPosition()
    if tip is not rig:
        tip.moveToGoodPosition()
    wired = node.input(1) is None
    if wired:
        node.setInput(1, tip)
    tip.setCurrent(True, clear_all_selected=True)
    common.say(
        node,
        "Created a pose rig%s. Pose and keyframe it, then add pose keys."
        % (" wired to input 1" if wired else "; wire it into input 1"),
        hou.severityType.ImportantMessage,
    )


def open_timeline(node) -> None:
    """Focus an existing Motion Timeline pane tab or float a new one."""
    node.setSelected(True, clear_all_selected=True)
    for tab in hou.ui.paneTabs():  # includes floating panels
        if tab.type() == hou.paneTabType.PythonPanel:
            iface = tab.activeInterface()
            if iface is not None and iface.name() == "fxmotion_timeline":
                tab.setIsCurrentTab()
                return
    iface = hou.pypanel.interfaceByName("fxmotion_timeline")
    if iface is None:
        common.fail(
            node,
            "Motion Timeline panel not found: is houdini/python_panels on "
            "HOUDINI_PATH (fxhoudinimotion package)?",
        )
        return
    tab = hou.ui.curDesktop().createFloatingPaneTab(hou.paneTabType.PythonPanel, size=(1100, 420))
    tab.setActiveInterface(iface)
    tab.setIsCurrentTab()
```

- [ ] **Step 5: Run the tests**

Run: `python -m pytest -q`
Expected: all pass, including the three panel-convention tests deselected in Task 8.

- [ ] **Step 6: Lint and commit**

```bash
uvx ruff@0.16.7 check --fix . && uvx ruff@0.16.7 format . && uvx ruff@0.16.7 check --fix . && uvx ruff@0.16.7 format . && uvx ruff@0.16.7 check . && uvx ruff@0.16.7 format --check .
git add houdini/python/fxmotion/nodes tests/test_nodes_source.py
git commit -m "feat(fxmotion): move the node logic out of the asset into fxmotion.nodes

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 10: build vb::kimodo_motion::2.0 and prove it matches 1.1

**Files:**
- Create: `scripts/build_hda.py`, `houdini/otls/vb_kimodo_motion_2.0.hda/` (generated, tracked)
- Move: `scripts/kimodo_icon.svg` -> `houdini/config/Icons/kimodo_motion.svg`
- Delete: `scripts/create_hda.py`, `scripts/_add_help.py`
- Modify: `tests/test_houdini_live.py`

**Interfaces:**
- Consumes: every callback name in `fxmotion.nodes.kimodo` and the four cooks in `fxmotion.nodes.common` (Task 9), the parm names listed in Task 9's interfaces.
- Produces: the node type `vb::kimodo_motion::2.0` (inputs: 0 Root Path, 1 Pose; outputs: 0 Rest Geometry, 1 Capture Pose, 2 Animated Pose, 3 T-Pose; TAB submenu "fxmotion"; icon `kimodo_motion.svg`).

- [ ] **Step 1: Write the failing live tests**

Append to `tests/test_houdini_live.py`, above `def run():`:

```python
def _repo():
    from pathlib import Path

    return Path(hou.text.expandString("$FXMOTION_ROOT"))


def test_kimodo_2_matches_1_1():
    """Same Kimodo clip, same root-path transform: output 2 of 2.0 must
    carry 1.1's positions, transforms and local transforms."""
    import sys
    import tempfile
    from pathlib import Path

    import numpy as np

    sys.path.insert(0, str(_repo() / "server"))
    import kimodo_adapter as ka

    from fxmotion import clipformat

    fixture = _repo() / "tests" / "fixtures" / "kimodo_stop.npz"
    geo = hou.node("/obj").createNode("geo", "fxmotion_parity")
    try:
        old = geo.createNode("vb::kimodo_motion::1.1")
        old.parm("npz_path").set(fixture.as_posix())
        old.parmTuple("path_xform").set((1.5, -2.0, 0.7))
        with np.load(fixture) as z:
            npz = {k: z[k] for k in z.files}
        path = Path(tempfile.mkdtemp()) / "parity.npz"
        clipformat.save(path, ka.to_clip(npz, ka.Canon(1.5, -2.0, 0.7)))
        new = geo.createNode("vb::kimodo_motion::2.0")
        new.parm("clip_path").set(path.as_posix())
        for n in (old, new):
            n.parm("start_frame").set(1)
            n.parm("retime").set(0)
        for frame in (1, 11, 26, 50):
            hou.setFrame(frame)
            a, b = old.geometry(2), new.geometry(2)
            assert a.pointStringAttribValues("name") == b.pointStringAttribValues("name")
            for attr, tol in (("P", 1e-5), ("transform", 1e-5), ("localtransform", 1e-4)):
                x = np.array(a.pointFloatAttribValues(attr))
                y = np.array(b.pointFloatAttribValues(attr))
                err = float(np.abs(x - y).max())
                assert err < tol, "%s differs by %g at frame %d" % (attr, err, frame)
    finally:
        geo.destroy()


def test_kimodo_2_outputs_and_details():
    import sys
    import tempfile
    from pathlib import Path

    import numpy as np

    sys.path.insert(0, str(_repo() / "server"))
    import kimodo_adapter as ka

    from fxmotion import clipformat

    fixture = _repo() / "tests" / "fixtures" / "kimodo_stop.npz"
    geo = hou.node("/obj").createNode("geo", "fxmotion_outputs")
    try:
        with np.load(fixture) as z:
            npz = {k: z[k] for k in z.files}
        path = Path(tempfile.mkdtemp()) / "out.npz"
        clipformat.save(path, ka.to_clip(npz, ka.Canon()))
        node = geo.createNode("vb::kimodo_motion::2.0")
        node.parm("clip_path").set(path.as_posix())
        counts = [len(node.geometry(i).points()) for i in range(4)]
        assert counts[0] > 1000 and counts[1] == counts[2] == counts[3] == 77, counts
        g = node.geometry(2)
        assert g.attribValue("fxmotion_skeleton") == "soma77"
        assert g.attribValue("fxmotion_fps") == 30.0
        assert '"backend": "kimodo"' in g.attribValue("fxmotion_source")
        assert g.findPointAttrib("contact") is not None
        assert node.errors() == ()
        node.parm("last_error").set("boom")
        try:
            node.cook(force=True)
        except hou.OperationFailed:
            pass  # cook() raises on a node in error, which is the point
        assert any("boom" in e for e in node.errors())
    finally:
        geo.destroy()
```

These fail until the asset exists (`hou.OperationFailed: Invalid node type name`).

- [ ] **Step 2: Move the icon and write the builder**

```bash
git mv scripts/kimodo_icon.svg houdini/config/Icons/kimodo_motion.svg
```

`scripts/build_hda.py`:

```python
"""Build vb::kimodo_motion::2.0 with hython, expanded into houdini/otls/.

    hython scripts/build_hda.py

Every callback and cook is one line into fxmotion.nodes: the logic lives in
the library (version-controlled, tested), the asset only wires parms to it.
The 1.1 asset in houdini/otls/ is not touched.
"""

import re
import shutil
import tempfile
from pathlib import Path

import hou

REPO = Path(__file__).resolve().parents[1]
OTLS = REPO / "houdini" / "otls"
NAME, VERSION, LABEL = "vb::kimodo_motion::2.0", "2.0", "Kimodo Motion"
LIBRARY = OTLS / "vb_kimodo_motion_2.0.hda"
SKELETON = "soma77"
MODELS = ("Kimodo-SOMA-RP-v1.1", "Kimodo-SOMA-SEED-v1.1", "Kimodo-SOMA-RP-v1")
PY = hou.scriptLanguage.Python
INDEX = "int(kwargs['script_multiparm_index']) - 1"
TIMELINE_OWNS = "{ has_timeline == 1 }"

PROMPT_HELP = (
    "What the character does, in plain __English__. Be specific about body "
    "part, direction, speed and style.\n\n"
    "__Name the body mechanics, not the intent.__ Measured: _a person jumps "
    "forward and lands on both feet_ gets both feet 0.22 off the ground; _a "
    "person leaps high into the air with both feet off the ground_ gets 0.94. "
    "Same model, same duration, 4.3x the result.\n\n"
    "__Do not prompt for finger or hand detail.__ Kimodo predicts on a 30-joint "
    "skeleton; the fingers you get back are reconstructed, never generated."
)

HELP = """= Kimodo Motion =

#type: node
#context: sop
#tags: kimodo, fxmotion, motion, ai, kinefx, animation

Generates human motion from text prompts with NVIDIA Kimodo, as a 77-joint
SOMA skeleton and a skinned body, ready for KineFX.

== Overview ==

The node talks to a running Kimodo server (see the fxhoudinimotion setup
guide). __Generate__ sends the segments, the root path on input 0 and the
pose keys from input 1; the finished clip downloads into __Download Dir__ and
the skeleton is rebuilt on every frame. Houdini stays interactive meanwhile.

Wire a __Joint Deform__ straight across (0 -> 0, 1 -> 1, 2 -> 2) for a moving
body, the same output order as Houdini's Test Geometry characters.

@inputs

Root Path:
    A curve or points: the root passes through their XZ positions. Points
    with an int `frame` attribute are timed waypoints.

Pose:
    A posed SOMA77 skeleton (see __Create Pose Rig__), sampled at the pose
    keys.

@outputs

Rest Geometry:
    The SOMA77 body, bound to the Capture Pose.

Capture Pose:
    The skeleton the body is bound to.

Animated Pose:
    The generated motion. Detail attributes `fxmotion_skeleton`,
    `fxmotion_fps` and `fxmotion_source` describe it.

T-Pose:
    The SOMA77 rest pose.
"""


def cb(fn, *args):
    extra = "".join(", %s" % a for a in args)
    return "from fxmotion.nodes import kimodo; kimodo.%s(kwargs['node']%s)" % (fn, extra)


def button(name, label, fn, *args, **kw):
    return hou.ButtonParmTemplate(name, label, script_callback=cb(fn, *args), script_callback_language=PY, **kw)


def hidden_string(name, label, **kw):
    return hou.StringParmTemplate(name, label, 1, default_value=("",), is_hidden=True, **kw)


def parms():
    ptg = hou.ParmTemplateGroup()

    gen = hou.FolderParmTemplate("fld_generate", "Generate", folder_type=hou.folderType.Tabs)
    gen.addParmTemplate(button("open_timeline", "Open Timeline", "open_timeline", is_label_hidden=True, join_with_next=True,
                               help="Open the __Motion Timeline__ panel for this node: prompt segments, transitions and pose tracks."))
    gen.addParmTemplate(button("generate", "Generate", "generate", is_label_hidden=True, join_with_next=True,
                               help="Send the request to the server. Progress shows in __Status__ and under the node."))
    gen.addParmTemplate(button("cancel", "Cancel", "cancel", is_label_hidden=True, help="Cancel the queued job or discard the running one."))
    gen.addParmTemplate(hou.MenuParmTemplate("model", "Model", MODELS, default_value=0,
                                             help="__RP__ = Bones Rigplay 1 (~700 h of mocap), the recommended default. __SEED__ = BONES-SEED (288 h, public data)."))
    gen.addParmTemplate(hou.ToggleParmTemplate("force", "Force Regenerate", default_value=True,
                                               help="Run inference again even if the server has an identical request cached."))
    gen.addParmTemplate(hou.IntParmTemplate("seed", "Seed", 1, default_value=(-1,), min=-1, max=100000, min_is_strict=True,
                                            help="`-1` = a new random result each time; any other value repeats a result."))
    seg = hou.FolderParmTemplate("segments", "Segments", folder_type=hou.folderType.ScrollingMultiparmBlock)
    seg.setDefaultValue(1)
    sync = cb("sync_segments")
    seg.addParmTemplate(hou.StringParmTemplate("seg_prompt#", "Prompt", 1, default_value=("",), script_callback=sync, script_callback_language=PY, help=PROMPT_HELP))
    seg.addParmTemplate(hou.IntParmTemplate("seg_from#", "Frames", 1, default_value=(0,), is_hidden=True))
    seg.addParmTemplate(hou.IntParmTemplate("seg_to#", "to", 1, default_value=(0,), is_hidden=True))
    seg.addParmTemplate(hou.LabelParmTemplate("seg_range#", "Frames", join_with_next=True,
                                              column_labels=('`chs("seg_from#")` - `chs("seg_to#")`   (`rint(ch("seg_frames#") / ch("scene_fps") * 100) / 100` s)',) + ("",) * 15))
    seg.addParmTemplate(hou.IntParmTemplate("seg_frames#", "Length", 1, default_value=(48,), min=1, max=240, min_is_strict=True,
                                            join_with_next=True, script_callback=sync, script_callback_language=PY, help="Length of this segment in scene frames."))
    seg.addParmTemplate(button("seg_split#", "Split", "split_segment", INDEX, join_with_next=True, help="Cut this segment in two at the playhead."))
    seg.addParmTemplate(button("seg_regen#", "Regenerate", "regenerate", INDEX, "False", join_with_next=True,
                               help="Re-roll this segment alone; the clip keeps its length and both joins stay continuous."))
    seg.addParmTemplate(button("seg_regen_end#", "From Here", "regenerate", INDEX, "True",
                               help="Re-roll this segment and every segment after it."))
    gen.addParmTemplate(seg)
    gen.addParmTemplate(hou.StringParmTemplate("prompt", "Prompt", 1, default_value=("a person walks forward",), is_hidden=True))
    gen.addParmTemplate(hou.IntParmTemplate("duration_frames", "Duration (frames)", 1, default_value=(72,), is_hidden=True))
    gen.addParmTemplate(hidden_string("status", "Status"))
    gen.addParmTemplate(hidden_string("clip_info", "Clip"))
    gen.addParmTemplate(hou.FloatParmTemplate("scene_fps", "Scene FPS", 1, default_expression=("$FPS",),
                                              default_expression_language=(hou.scriptLanguage.Hscript,), is_hidden=True))
    ptg.append(gen)

    con = hou.FolderParmTemplate("fld_constraints", "Constraints", folder_type=hou.folderType.Tabs)
    path = hou.FolderParmTemplate("grp_path", "Root Path (input 0)", folder_type=hou.folderType.Collapsible, tags={"group_default": "1"})
    path.addParmTemplate(hou.IntParmTemplate("path_waypoints", "Path Waypoints", 1, default_value=(8,), min=0, max=64, min_is_strict=True,
                                             help="The curve is thinned to this many points by arc length. `0` = every point."))
    con.addParmTemplate(path)
    js = hou.FolderParmTemplate("grp_json", "Constraints JSON", folder_type=hou.folderType.Collapsible, tags={"group_default": "0"})
    js.addParmTemplate(hou.StringParmTemplate("constraints_file", "Constraints File", 1, default_value=("",), string_type=hou.stringParmType.FileReference,
                                              file_type=hou.fileType.Any, tags={"filechooser_pattern": "*.json"},
                                              help="Kimodo constraints JSON, in Kimodo's own model space. Ignored when Constraints JSON is set."))
    js.addParmTemplate(hou.StringParmTemplate("constraints_json", "Constraints JSON", 1, default_value=("",), tags={"editor": "1", "editorlines": "3-8"},
                                              help="Inline Kimodo constraints JSON (a list of constraint dicts), in Kimodo's model space."))
    con.addParmTemplate(js)
    pose = hou.FolderParmTemplate("grp_pose", "Pose Keyframes (input 1)", folder_type=hou.folderType.Collapsible, tags={"group_default": "1"})
    pose.addParmTemplate(button("make_pose_rig", "Create Pose Rig", "make_pose_rig", help="Drop a capture-pose rig wired to input 1, ready to pose."))
    pose.addParmTemplate(hou.StringParmTemplate("pose_keyframes", "Pose Keyframes", 1, default_value=("",), disable_when=TIMELINE_OWNS,
                                                help="Scene frames to sample input 1 at, e.g. `1 45 89`. Disabled while the timeline owns the keys."))
    pose.addParmTemplate(hou.MenuParmTemplate("pose_type", "Pose Constraint", ("Full-Body", "End-Effector"), default_value=0,
                                              disable_when='{ pose_keyframes == "" } ' + TIMELINE_OWNS))
    ee = '{ pose_type != "End-Effector" } { pose_keyframes == "" } ' + TIMELINE_OWNS
    for name, label, on, join in (("ee_left_hand", "Left Hand", False, True), ("ee_right_hand", "Right Hand", True, False),
                                  ("ee_left_foot", "Left Foot", False, True), ("ee_right_foot", "Right Foot", False, False)):
        pose.addParmTemplate(hou.ToggleParmTemplate(name, label, default_value=on, disable_when=ee, join_with_next=join))
    con.addParmTemplate(pose)
    ptg.append(con)

    out = hou.FolderParmTemplate("fld_output", "Output", folder_type=hou.folderType.Tabs)
    out.addParmTemplate(hou.IntParmTemplate("start_frame", "Start Frame", 1, script_callback=cb("refresh_starts"), script_callback_language=PY,
                                            default_expression=("$FSTART",), default_expression_language=(hou.scriptLanguage.Hscript,),
                                            min=-1000, max=1000, help="Scene frame on which the clip begins."))
    out.addParmTemplate(hou.StringParmTemplate("clip_path", "Clip Path", 1, default_value=("",), string_type=hou.stringParmType.FileReference,
                                               file_type=hou.fileType.Any, tags={"filechooser_pattern": "*.npz"},
                                               help="The fxmotion clip the node reads. Set by Generate; any fxmotion.clip/1 file works without a server."))
    adv = hou.FolderParmTemplate("grp_advanced", "Advanced", folder_type=hou.folderType.Collapsible, tags={"group_default": "0"})
    adv.addParmTemplate(hou.ToggleParmTemplate("retime", "Retime to Scene FPS", default_value=True,
                                               help="Keep the clip's real duration at any `$FPS` (nearest sample). Off = one clip sample per scene frame."))
    out.addParmTemplate(adv)
    ptg.append(out)

    srv = hou.FolderParmTemplate("fld_server", "Server", folder_type=hou.folderType.Tabs)
    srv.addParmTemplate(hou.StringParmTemplate("server_url", "API Server URL", 1, default_value=("http://localhost:8001",), join_with_next=True))
    srv.addParmTemplate(button("test_connection", "Test Connection", "test_connection"))
    srv.addParmTemplate(hou.StringParmTemplate("download_dir", "Download Dir", 1, default_value=("$HIP/fxmotion_cache",),
                                               string_type=hou.stringParmType.FileReference, file_type=hou.fileType.Directory))
    ptg.append(srv)

    for name, label in (("job_id", "Job ID"), ("last_error", "Last Error")):
        ptg.append(hidden_string(name, label))
    ptg.append(hou.FloatParmTemplate("progress", "Progress", 1, default_value=(0.0,), min=0.0, max=1.0, is_hidden=True))
    ptg.append(hidden_string("timeline_json", "Timeline", tags={"editor": "1"}))
    ptg.append(hou.ToggleParmTemplate("has_timeline", "Has Timeline", default_value=False, is_hidden=True))
    ptg.append(hou.IntParmTemplate("frame_ref", "Frame", 1, default_expression=("$F",),
                                   default_expression_language=(hou.scriptLanguage.Hscript,), is_hidden=True))
    return ptg


def _patch_dialog_script(definition, labels):
    """Input/output connector labels and the Segments minimum live only in
    the DialogScript."""
    ds = definition.sections()["DialogScript"].contents().splitlines(keepends=True)
    inputs = {"1": "Root Path / Waypoints (opt)", "2": "Pose / skeleton (opt)"}

    def relabel(line):
        s = line.lstrip()
        for n, lbl in inputs.items():
            if s.startswith(("inputlabel\t%s" % n, "inputlabel %s" % n)):
                return '    inputlabel\t%s\t"%s"\n' % (n, lbl)
        return line

    ds = [relabel(line) for line in ds]
    for i, line in enumerate(ds):
        if line.strip() == 'name    "segments"':
            for j in range(i, min(i + 6, len(ds))):
                if ds[j].lstrip().startswith("default"):
                    indent = ds[j][: len(ds[j]) - len(ds[j].lstrip())]
                    ds.insert(j + 1, indent + "range   { 1! 100 }\n")
                    break
            break
    after = max(i for i, line in enumerate(ds) if line.lstrip().startswith("inputlabel"))
    inject = "".join('    outputlabel\t%d\t"%s"\n' % (i + 1, lbl) for i, lbl in enumerate(labels))
    definition.addSection("DialogScript", "".join(ds[: after + 1]) + inject + "".join(ds[after + 1 :]))


def build():
    geo = hou.node("/obj").createNode("geo", "kimodo_motion_build")
    try:
        subnet = geo.createNode("subnet", "kimodo_motion")
        cooks = (
            ("rest_geometry", "common.cook_file(hou.pwd(), %r, 'skin')" % SKELETON),
            ("capture_pose", "common.cook_file(hou.pwd(), %r, 'capture_pose')" % SKELETON),
            ("animated_pose", "common.cook_animated(hou.pwd())"),
            ("t_pose", "common.cook_rest(hou.pwd(), %r)" % SKELETON),
        )
        colors = {0: (0.584, 0.776, 1.0), 2: (0.976, 0.780, 0.263)}  # as kinefx::characterio::2.0
        for idx, (name, call) in enumerate(cooks):
            sop = subnet.createNode("python", name)
            sop.parm("python").set("from fxmotion.nodes import common\n%s\n" % call)
            out = subnet.createNode("output", "output%d" % idx)
            out.setInput(0, sop)
            out.parm("outputidx").set(idx)
            if idx in colors:
                out.setColor(hou.Color(colors[idx]))
            if idx == 0:
                out.setDisplayFlag(True)
                out.setRenderFlag(True)
        subnet.layoutChildren()

        packed = Path(tempfile.mkdtemp()) / "vb_kimodo_motion_2.0.hda"
        node = subnet.createDigitalAsset(
            name=NAME, hda_file_name=str(packed), description=LABEL,
            min_num_inputs=0, max_num_inputs=2, version=VERSION,
        )
        d = node.type().definition()
        d.setMaxNumOutputs(len(cooks))
        d.setIcon("kimodo_motion.svg")
        d.setParmTemplateGroup(parms())
        d.addSection("OnCreated", "from fxmotion.nodes import kimodo\nkimodo.on_created(kwargs['node'])\n")
        d.setExtraFileOption("OnCreated/IsPython", True)
        d.addSection("DescriptiveParmName", "status")
        d.addSection("Help", HELP)
        shelf = d.sections().get("Tools.shelf")
        if shelf is not None:
            d.addSection("Tools.shelf", re.sub(r"<toolSubmenu>.*?</toolSubmenu>", "<toolSubmenu>fxmotion</toolSubmenu>", shelf.contents(), count=1))
        d.save(str(packed))
        _patch_dialog_script(d, ["Rest Geometry", "Capture Pose", "Animated Pose", "T-Pose"])
        d.save(str(packed))
        if LIBRARY.exists():
            shutil.rmtree(LIBRARY)
        hou.hda.expandToDirectory(str(packed), str(LIBRARY))
        print("HDA saved: %s  type: %s" % (LIBRARY, NAME))
    finally:
        geo.destroy()


build()
```

- [ ] **Step 3: Build the asset and delete the old builder**

```bash
hython scripts/build_hda.py
git rm scripts/create_hda.py scripts/_add_help.py
git status --short houdini/otls
```

Expected: `HDA saved: .../houdini/otls/vb_kimodo_motion_2.0.hda  type: vb::kimodo_motion::2.0`, a new untracked `houdini/otls/vb_kimodo_motion_2.0.hda/` directory, and no change under `vb_kimodo_motion_1.1.hda/`. If `hython` is not on PATH, run it as `"$HFS/bin/hython"`.

- [ ] **Step 4: Run the live tests in Houdini**

In a running Houdini session with the fxhoudinimotion package loaded (`FXMOTION_ROOT` set, see Task 11 Step 1; restart Houdini after editing the package file, then reload the asset library with `hou.hda.reloadAllFiles()`), in the Python Shell:

```python
exec(open(hou.text.expandString("$FXMOTION_ROOT/tests/test_houdini_live.py")).read()); print("\n".join(run()))
```

Expected: every line starts with `ok`, including `ok   test_kimodo_2_matches_1_1` and `ok   test_kimodo_2_outputs_and_details`. If parity fails, the 1.1 node is the reference: fix 2.0 (the adapter or `clip.joint_frames`), never `_reference_v11.py`.

- [ ] **Step 5: Run the offline suite, lint and commit**

```bash
python -m pytest -q
uvx ruff@0.16.7 check --fix . && uvx ruff@0.16.7 format . && uvx ruff@0.16.7 check --fix . && uvx ruff@0.16.7 format . && uvx ruff@0.16.7 check . && uvx ruff@0.16.7 format --check .
git add scripts/build_hda.py houdini/otls/vb_kimodo_motion_2.0.hda houdini/config/Icons/kimodo_motion.svg tests/test_houdini_live.py
git add -u scripts
git commit -m "feat(hda): build kimodo_motion 2.0 on fxmotion, matching 1.1 output

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 11: package variable, docs and the spec addendum

**Files:**
- Modify: `fxhoudinimotion.json`, `README.md`, `docs/setup.md`, `houdini/README.md`, `CONTRIBUTING.md`, `superpowers/specs/2026-09-29-fxhoudinimotion-design.md`

- [ ] **Step 1: Rename the package variable**

Replace `fxhoudinimotion.json` with:

```json
{
    "env": [
        {
            "FXMOTION_ROOT": "/path/to/fxhoudinimotion"
        },
        {
            "PYTHONPATH": {
                "method": "append",
                "value": "$FXMOTION_ROOT/houdini/python"
            }
        },
        {
            "PYTHONPATH": {
                "method": "append",
                "value": "$FXMOTION_ROOT/vendor"
            }
        }
    ],
    "path": "$FXMOTION_ROOT/houdini"
}
```

Run: `git grep -n "KIMODO_BRIDGE_ROOT" -- . ':!houdini/otls/vb_kimodo_motion_1.1.hda'`
Expected after Step 2: no output.

- [ ] **Step 2: Update the docs**

In `README.md`, `docs/setup.md`, `houdini/README.md` and `CONTRIBUTING.md`:
- `KIMODO_BRIDGE_ROOT` -> `FXMOTION_ROOT` everywhere.
- The server deployment steps: instead of copying `kimodo_server.py` into the kimodo checkout, copy only `docker-compose.bridge.yaml` and set `FXMOTION_ROOT` in the kimodo checkout's `.env` (next to `HF_HOME`), e.g. `FXMOTION_ROOT=C:/Users/<you>/Documents/GitHub/fxhoudinimotion`; the container mounts the repo and runs `kimodo_backend`.
- The node: `vb::kimodo_motion::2.0`, parm **Clip Path** (was NPZ Path), no Clip FPS parm, new **Seed**, TAB submenu **fxmotion**, the panel is **Motion Timeline**. State that 1.1 nodes still play the clips they already downloaded but cannot Generate against the new server; recreate the node as 2.0 to generate.
- Rebuilding the asset: `hython scripts/build_hda.py` (replaces `create_hda.py` + `_add_help.py`).
- In `README.md`'s Architecture diagram and Environment Variables table, add `FXMOTION_IDLE_UNLOAD_S` (seconds before the server frees the model's VRAM, default 0 = never) and `FXMOTION_MOCK_CLIP` (Kimodo NPZ served in MOCK_MODE).

Run: `git grep -n -i "kimodo_server\|create_hda\|_add_help\|kimodo_timeline\|npz_path\|source_fps" -- . ':!houdini/otls/vb_kimodo_motion_1.1.hda' ':!superpowers' ':!tests/_reference_v11.py'`
Expected: no output.

- [ ] **Step 3: Record the clarifications in the spec**

Append to `superpowers/specs/2026-09-29-fxhoudinimotion-design.md`:

```markdown
## 10. Addendum: decisions made while planning phase 1 (2026-09-29)

The twelve "Spec clarifications decided while planning" of
`superpowers/plans/2026-09-29-phase1-foundation-kimodo.md` apply to this
spec: effectors folded into keyframes (`joints`), the `options` dict, the
`continue` capability, optional `time_s` on root-path points, server-side
canonicalisation (which fixes 1.1's pose-key / root-path mismatch), the floor
invariant on generated clips only, ORTHO_TOL = 1e-2, parent-first joint
order, 1.1 nodes playing but not generating, the repo mounted into the Kimodo
container, no `source_fps` parm, and pose keys without input 1 raising.
```

- [ ] **Step 4: Final checks and commit**

```bash
python -m pytest -q
uvx ruff@0.16.7 check --fix . && uvx ruff@0.16.7 format . && uvx ruff@0.16.7 check . && uvx ruff@0.16.7 format --check .
git add fxhoudinimotion.json README.md docs/setup.md houdini/README.md CONTRIBUTING.md superpowers/specs/2026-09-29-fxhoudinimotion-design.md
git commit -m "docs: move setup to FXMOTION_ROOT, the mounted server and kimodo_motion 2.0

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

Then, with Docker running, a real end-to-end check (not automatable in CI): start the stack with `MOCK_MODE=0`, create a `vb::kimodo_motion::2.0` node, Generate a two-segment timeline with a curve on input 0, and confirm the body walks along the curve. Report the result in the PR description, including a failure if there is one.
