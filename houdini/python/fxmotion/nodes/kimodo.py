"""The Kimodo Motion node: its parms and inputs as a request."""

from __future__ import annotations

import json

import hou
import numpy as np

from .. import client, paths, skeletons
from . import common, timeline_parms
from .common import cancel, test_connection  # noqa: F401  (HDA callbacks)
from .timeline_parms import (  # noqa: F401  (HDA callbacks)
    on_created,
    refresh_starts,
    split_segment,
)

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
                "%s: generation started, watch the node's Status."
                % common.label(node),
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
            {
                "prompt": s["prompt"].strip(),
                "duration_s": int(s["frames"]) / scene_fps,
            }
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
            (
                None if track == "fullbody" else [track],
                sorted(int(k) for k in keys),
            )
            for track, keys in (tl.get("tracks") or {}).items()
            if keys
        ]
    else:
        raw = node.parm("pose_keyframes").eval().replace(",", " ")
        frames = [int(x) for x in raw.split()]
        groups = []
        if frames:
            joints = None
            if node.parm("pose_type").evalAsString() == "End-Effector":
                joints = [
                    j for parm, j in EFFECTOR_PARMS if node.parm(parm).eval()
                ]
                if not joints:
                    raise ValueError(
                        "End-Effector pose constraint: select at least one "
                        "hand or foot."
                    )
            groups = [(joints, frames)]
    if not groups:
        return []
    ins = node.inputs()
    if len(ins) < 2 or ins[1] is None:
        raise ValueError(
            "Pose keys are set but nothing is wired to input 1 (posed "
            "skeleton)."
        )
    out = []
    for joints, frames in groups:
        for f in frames:
            pos, rot = _pose_at(ins[1], f)
            out.append(
                {
                    "time_s": secs(f),
                    "world_pos": pos,
                    "world_rot": rot,
                    "joints": joints,
                }
            )
    return out


def _pose_at(src, frame):
    """World positions and KineFX transforms of every SOMA77 joint on the
    posed skeleton `src`, at a scene frame. The server inverts the
    transform, so nothing is converted here."""
    skel = skeletons.get(SKELETON)
    index = {n: i for i, n in enumerate(skel.joint_names)}
    geo = src.geometryAtFrame(frame)
    if (
        geo.findPointAttrib("name") is None
        or geo.findPointAttrib("transform") is None
    ):
        raise ValueError(
            "Input 1 must be a SOMA77 skeleton with name and transform point "
            "attributes."
        )
    names = geo.pointStringAttribValues("name")
    pos_all = np.asarray(geo.pointFloatAttribValues("P")).reshape(-1, 3)
    rot_all = np.asarray(geo.pointFloatAttribValues("transform")).reshape(
        -1, 3, 3
    )
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
    tab = hou.ui.curDesktop().createFloatingPaneTab(
        hou.paneTabType.PythonPanel, size=(1100, 420)
    )
    tab.setActiveInterface(iface)
    tab.setIsCurrentTab()
