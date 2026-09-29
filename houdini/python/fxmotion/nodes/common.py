"""Generator-node plumbing shared by every model: jobs, errors and cooks."""

from __future__ import annotations

import hou

from .. import client, clip, clipformat, skeletons


def url(node) -> str:
    return node.parm("server_url").eval().rstrip("/")


def label(node) -> str:
    return node.type().description()


def say(node, msg, severity=None) -> None:
    """Status parm (shown under the node and in the panel) plus the status
    bar."""
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
            path = client.download(
                url(node), job_id, node.parm("download_dir").eval()
            )
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
        "Server OK: %s, %s at %g fps%s"
        % (h["backend"], h["skeleton"], h["fps"], mock),
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
    try:
        c = clip.load(path)
    except (clipformat.ClipError, OSError) as e:
        # a wrong or missing file is the user's to fix: say why, not where
        raise hou.NodeError("Clip Path %s: %s" % (path, e)) from None
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
        raise hou.NodeError(
            "skeleton %s has no %s geometry" % (skeleton, which)
        )
    sop.geometry().loadFromFile(str(path))


def cook_file(sop, skeleton, which) -> None:
    raise_last_error(sop.parent())
    load_skeleton_file(sop, skeleton, which)
