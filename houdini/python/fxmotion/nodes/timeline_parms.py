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
    start = int(node.parm("start_frame").eval()) + sum(
        int(s["frames"]) for s in segs[:index]
    )
    left = int(round(hou.frame())) - start
    if 0 < left < int(segs[index]["frames"]):
        segs.insert(
            index + 1,
            {
                "prompt": segs[index]["prompt"],
                "frames": int(segs[index]["frames"]) - left,
            },
        )
        segs[index]["frames"] = left
        tl["segments"] = segs
        with hou.undos.group("%s: split segment" % common.label(node)):
            write_timeline(node, tl)
            rebuild_segments(node)
    elif hou.isUIAvailable():
        hou.ui.setStatusMessage(
            "%s: put the playhead inside this segment to split it."
            % common.label(node),
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
                "%s: %s" % (common.label(node), e),
                severity=hou.severityType.Warning,
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
            node.parm("seg_frames1").set(
                int(node.parm("duration_frames").eval())
            )
        sync_from_parms(node)
