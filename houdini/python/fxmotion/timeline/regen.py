"""Regenerate the tail of a generated clip instead of the whole thing.

Kimodo builds a multi-prompt clip one segment at a time, each segment joined to
the previous by a transition: the previous tail is prepended as observed motion,
moved to the origin, generated against, then moved back and alpha-blended (see
`_multiprompt` in kimodo/model/kimodo_model.py).

A plain /generate always starts with that history empty, so its first segment
takes the "first motion" path and the transition never runs. Handing the tail
over as a world-space constraint does not substitute for it: the constrained
frames come back accurate to a millimetre, but the motion after them continues
on its own trajectory, measured 184 cm away.

So the server takes a `continue_from` tail and seeds that history instead, which
makes the first requested segment join properly. Measured on a 545-sample clip,
the seam moves 2.18 cm against the clip's own 10.48 cm mean per sample, so the
join is smoother than ordinary motion.

The returned clip's head IS the blended seam, so the merge follows the model's
own convention, old[:cut - n] + new, with no frames dropped.

The pure functions take plain data so they can be tested without Houdini.
"""

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


class Precondition(ValueError):
    """Something the user can fix and ask again, not a failure: which button
    to press first, what to update. Callers show these as a warning and leave
    last_error alone, so the node does not go red over a wrong turn."""


def cut_sample(
    frames_per_segment, seg_index: int, scene_fps: float, source_fps: float
) -> int:
    """Clip sample where `seg_index` begins. Lengths are in scene frames.

    Summed per segment rather than converted from the cumulative scene frame,
    because that is how the clip was laid out: the server truncates each
    segment on its own (`int(duration * fps)`, see `samples_for`), so rounding
    the running total instead disagrees with it by a sample on about a third
    of segment lengths at 24 fps against a 30 fps model. That sample is the
    difference between the tail pin landing on the join and landing next to it.
    """
    return sum(
        samples_for(int(f), scene_fps, source_fps)
        for f in frames_per_segment[:seg_index]
    )


def continue_payload(clip, cut: int, n: int) -> dict:
    """The `n` samples before `cut`, in Kimodo's model space, as the
    request's `continue_from` block. Local rotations rather than global:
    they are what the motion representation is built from, and the server's
    77 -> model-skeleton slice is the exact inverse of the output
    conversion, so the round trip is lossless."""
    if cut < n:
        raise ValueError(
            "need %d samples before the cut, cut is at %d" % (n, cut)
        )
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
    d = np.linalg.norm(np.diff(clip["world_pos"], axis=0), axis=-1)
    return float(d.max(axis=1).mean())


def tail_pin(clip, cut_end: int, n: int, at_index: int) -> list:
    """Constraints holding the new segment's last `n` frames to the `n`
    samples before `cut_end`, in model space (Kimodo native constraint
    dicts), so whatever follows in the existing clip still joins on.
    Indices are segment-relative: the model crops user constraints with
    current_frame = 0 for the first requested segment."""
    head = slice(cut_end - n, cut_end)
    root = clip["native_smooth_root_pos"][head][:, [0, 2]]
    common = {
        "frame_indices": list(range(at_index, at_index + n)),
        "global_joints_positions": clip["native_posed_joints"][head].tolist(),
        "global_joints_rots": clip["native_global_rot_mats"][head].tolist(),
        "smooth_root_2d": root.tolist(),
    }
    return [
        dict(common, type="fullbody-global"),
        dict(
            common,
            type="ee-global",
            joint_names=["LeftHand", "RightHand", "LeftFoot", "RightFoot"],
        ),
    ]


def samples_for(frames: int, scene_fps: float, source_fps: float) -> int:
    """Samples the server generates for a segment: it uses int(duration*fps)."""
    return max(1, int(frames / scene_fps * source_fps))


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
        raise Precondition(
            "Segment %d is outside the timeline." % (seg_index + 1)
        )
    if seg_index == 0:
        raise Precondition(
            "Segment 1 has no earlier motion to continue from; use Generate."
        )
    src = node.parm("clip_path").eval()
    if not src or not Path(src).exists():
        raise Precondition(
            "No generated clip on this node yet. Press Generate first."
        )
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
            "would land in the wrong place. Press Generate first."
            % (have, expect)
        )
    if cut - n < 1 or cut > have:
        raise Precondition(
            "Segment %d falls outside the clip on disk; press Generate."
            % (seg_index + 1)
        )

    single = not to_end and seg_index + 1 < len(tl.segments)
    send = (
        tl.segments[seg_index : seg_index + 1]
        if single
        else tl.segments[seg_index:]
    )
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

    what = (
        "segment %d" % (seg_index + 1)
        if single
        else "from segment %d" % (seg_index + 1)
    )

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
            joins = "%.2f / %.2f cm" % (jump * 100, tail * 100)
            worst = max(jump, tail)
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
