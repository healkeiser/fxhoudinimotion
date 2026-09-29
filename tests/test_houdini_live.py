"""Live-session test: drive the real panel widgets inside a running Houdini.

Run INSIDE Houdini (Python Shell or the MCP bridge), not hython: it needs hou.ui
and a real Qt application.

    exec(open("tests/test_houdini_live.py").read()); print("\n".join(run()))

SCOPE, honestly stated. These assert the hygiene properties we can measure: that
the segment dialog is never modal, that dialogs and menus do not accumulate,
that nothing waits on a nested event loop, that no Qt grab or modal widget is
left behind.

They still do NOT prove the "Houdini panes stop answering the mouse" wedge is
gone. We now know exactly what that wedge is: after a nested Qt event loop runs
inside Houdini's UI pump, every NATIVE mouse message is delivered to Houdini's
panes twice, so their press/release pairing never rebalances. Traced event by
event with scripts/diagnose_input_wedge.py. But only a native message doubles; a
synthetic one sent with sendEvent arrives exactly once, measured. So no test can
generate the condition, and a human still has to click a Houdini pane to
confirm. Use trace() in that script.
"""

import contextlib
import gc

import hou

from fxmotion.qt import QtCore, QtWidgets
from fxmotion.timeline import model, widget


def _app():
    return QtWidgets.QApplication.instance()


def _flush_deletes():
    """Deliver DeferredDelete. processEvents() does NOT, so a hand-rolled flush
    that uses it alone reports objects as leaked when they are merely pending.
    Measured: one PromptDialog survives processEvents and dies on
    sendPostedEvents."""
    for _ in range(
        3
    ):  # deleting a widget posts further deletes for its children
        _app().processEvents()
        _app().sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
    gc.collect()


def _count(cls):
    _flush_deletes()
    return len([o for o in gc.get_objects() if isinstance(o, cls)])


def _answer_dialog(accept=True):
    """Answer the segment dialog _ask has just shown, in line.

    No timer, and that is the point: _ask no longer waits, so nothing here would
    pump a timer. accept() emits finished synchronously, which is what runs the
    callback.
    """
    for w in _app().topLevelWidgets():
        if isinstance(w, widget.PromptDialog) and w.isVisible():
            w.accept() if accept else w.reject()
            return True
    return False


def _close_visible(cls, delay_ms=60, tries=50):
    """Close the next visible widget of `cls`, from the event loop.

    Armed BEFORE the call that shows it, because contextMenuEvent execs the menu
    and does not return until it closes. exec pumps the event loop, which is
    what lets this timer fire at all. Bounded, so a menu that never appears
    cannot spin forever.
    """

    def go(n=0):
        for w in _app().topLevelWidgets():
            if isinstance(w, cls) and w.isVisible():
                w.close()
                return
        if n < tries:
            QtCore.QTimer.singleShot(delay_ms, lambda: go(n + 1))

    QtCore.QTimer.singleShot(delay_ms, go)


def _canvas():
    c = widget.Canvas()
    c.tl = model.Timeline([model.Segment("walk", 24)])
    return c


def test_segment_dialog_is_never_modal():
    """Modality is the property that pulls Houdini into a blocked state.
    Whatever the exact mechanism, the dialog has no business being modal."""
    dlg = widget.PromptDialog("x", 24, hou.qt.mainWindow())
    dlg.show()
    try:
        assert not dlg.isModal(), "the segment dialog is modal"
        assert _app().activeModalWidget() is None, "a modal widget is active"
    finally:
        dlg.close()
        dlg.deleteLater()


def test_ask_calls_back_without_waiting():
    """_ask hands its result to a callback and returns at once. If it ever went
    back to waiting for the answer, it would need a nested loop, and that is the
    wedge."""
    c = _canvas()
    got = []
    try:
        c._ask("walk", 24, lambda text, frames: got.append((text, frames)))
        assert not got, (
            "_ask blocked; it must return before the dialog is answered"
        )
        assert _answer_dialog(), "_ask showed no dialog"
        assert got == [("walk", 24)], "callback got %r" % (got,)
        assert _app().activeModalWidget() is None, "left a modal widget behind"
    finally:
        c.deleteLater()


def test_ask_ignores_a_rejected_dialog():
    """Cancel must change nothing."""
    c = _canvas()
    got = []
    try:
        c._ask("walk", 24, lambda text, frames: got.append((text, frames)))
        assert _answer_dialog(accept=False), "_ask showed no dialog"
        assert got == [], "Cancel still ran the callback: %r" % (got,)
    finally:
        c.deleteLater()


def test_dialogs_do_not_accumulate():
    """Parented to the main window, a dialog lives until deleted. Three edits
    must not leave three dialogs alive for the rest of the session."""
    c = _canvas()
    try:
        before = _count(
            widget.PromptDialog
        )  # delta, not absolute: other tests ran
        for _ in range(3):
            c._ask("walk", 24, lambda text, frames: None)
            assert _answer_dialog(), "_ask showed no dialog"
        after = _count(widget.PromptDialog)
        assert after <= before, "segment dialogs accumulating: %d -> %d" % (
            before,
            after,
        )
    finally:
        c.deleteLater()


def test_real_context_menu_does_not_leak():
    """Drives the actual contextMenuEvent, not a synthetic QMenu loop.

    An earlier version of this test built its own menus and reported one
    leaking; that was the loop variable still holding the last one at count
    time. Driving the real handler and letting the local fall out of scope is
    the only version that measures our code.
    """
    from fxmotion.qt import QtGui

    c = _canvas()
    c.resize(600, 200)
    try:
        before = _count(QtWidgets.QMenu)
        for _ in range(3):
            _close_visible(
                QtWidgets.QMenu
            )  # armed first: contextMenuEvent blocks
            ev = QtGui.QContextMenuEvent(
                QtGui.QContextMenuEvent.Mouse,
                QtCore.QPoint(100, 30),  # over a segment
                QtCore.QPoint(500, 500),
            )
            c.contextMenuEvent(ev)
        c._menu = None  # the panel holds the last one on purpose; release it
        assert _app().activePopupWidget() is None, (
            "context menu left a popup open"
        )
        after = _count(QtWidgets.QMenu)
        assert after <= before, "context menus accumulating: %d -> %d" % (
            before,
            after,
        )
    finally:
        c.deleteLater()


def test_no_input_grab_left_behind():
    """Whatever the wedge is, a Qt grab would be one explanation. Assert we
    never leave one, so that explanation can be ruled out rather than guessed
    at."""
    c = _canvas()
    try:
        c._ask("walk", 24, lambda text, frames: None)
        assert _answer_dialog(), "_ask showed no dialog"
        assert QtWidgets.QWidget.mouseGrabber() is None, (
            "mouse grab left behind"
        )
        assert QtWidgets.QWidget.keyboardGrabber() is None, (
            "keyboard grab left behind"
        )
        assert _app().activePopupWidget() is None, "popup left behind"
        assert _app().overrideCursor() is None, "override cursor left behind"
    finally:
        c.deleteLater()


def test_no_mouse_event_escapes_the_canvas():
    """The input wedge, as a test that needs no human.

    QApplication.notify walks a press up the parent chain for as long as each
    widget ignores it, and a synthetic press walks it exactly like a real one.
    So put the canvas inside a host widget and count who receives one press per
    button. Anything above the canvas means it would have reached Houdini's pane
    in a real panel, and a right press that gets there without its release is
    what wedges Houdini.
    """
    seen = []

    class Spy(QtCore.QObject):
        def eventFilter(self, obj, ev):
            if ev.type() in (
                QtCore.QEvent.MouseButtonPress,
                QtCore.QEvent.MouseButtonRelease,
            ):
                seen.append(
                    "%s.%s"
                    % (type(obj).__name__, ev.type().name.split("_")[-1])
                )
            return False

    from fxmotion.qt import QtGui

    host = QtWidgets.QWidget()  # stands in for Houdini's pane
    c = widget.Canvas(host)
    c.tl = model.Timeline([model.Segment("walk", 24)])
    c.resize(600, 200)
    spy = Spy()
    _app().installEventFilter(spy)
    try:
        for button in (
            QtCore.Qt.RightButton,
            QtCore.Qt.MiddleButton,
            QtCore.Qt.LeftButton,
        ):
            for kind in (
                QtCore.QEvent.MouseButtonPress,
                QtCore.QEvent.MouseButtonRelease,
            ):
                del seen[:]
                ev = QtGui.QMouseEvent(
                    kind,
                    QtCore.QPointF(50, 50),
                    button,
                    button,
                    QtCore.Qt.NoModifier,
                )
                _app().sendEvent(c, ev)
                escaped = [s for s in seen if not s.startswith("Canvas")]
                assert not escaped, (
                    "%s %s escaped the canvas to %s; in a panel that is Houdini's pane"
                    % (button, kind, escaped)
                )
    finally:
        _app().removeEventFilter(spy)
        c.deleteLater()
        host.deleteLater()


def _repo():
    """The clone: from FXMOTION_ROOT, else from where fxmotion was imported
    (this file is exec'd, so it has no __file__ of its own)."""
    import os
    from pathlib import Path

    import fxmotion

    root = os.environ.get("FXMOTION_ROOT")
    return Path(root) if root else Path(fxmotion.__file__).resolve().parents[3]


def _kimodo_clip(canon):
    """The Kimodo fixture as a saved fxmotion clip, via the server's own
    adapter."""
    import sys
    import tempfile
    from pathlib import Path

    import numpy as np

    from fxmotion import clipformat

    sys.path.insert(0, str(_repo() / "server"))
    import kimodo_adapter as ka

    fixture = _repo() / "tests" / "fixtures" / "kimodo_stop.npz"
    with np.load(fixture) as z:
        npz = {k: z[k] for k in z.files}
    path = Path(tempfile.mkdtemp()) / "clip.npz"
    clipformat.save(path, ka.to_clip(npz, ka.Canon(*canon)))
    return fixture, path


def test_kimodo_2_matches_1_1():
    """Same Kimodo clip, same root-path transform: output 2 of 2.0 must
    carry 1.1's positions, transforms and local transforms."""
    import numpy as np

    canon = (1.5, -2.0, 0.7)
    fixture, path = _kimodo_clip(canon)
    geo = hou.node("/obj").createNode("geo", "fxmotion_parity")
    try:
        old = geo.createNode("vb::kimodo_motion::1.1")
        old.parm("npz_path").set(fixture.as_posix())
        old.parmTuple("path_xform").set(canon)
        new = geo.createNode("vb::kimodo_motion::2.0")
        new.parm("clip_path").set(path.as_posix())
        for n in (old, new):
            n.parm("start_frame").set(1)
            n.parm("retime").set(0)
        for frame in (1, 11, 26, 50):
            hou.setFrame(frame)
            a, b = old.geometry(2), new.geometry(2)
            names_a = a.pointStringAttribValues("name")
            assert names_a == b.pointStringAttribValues("name")
            for attr, tol in (
                ("P", 1e-5),
                ("transform", 1e-5),
                ("localtransform", 1e-4),
            ):
                x = np.array(a.pointFloatAttribValues(attr))
                y = np.array(b.pointFloatAttribValues(attr))
                err = float(np.abs(x - y).max())
                assert err < tol, "%s differs by %g at frame %d" % (
                    attr,
                    err,
                    frame,
                )
    finally:
        geo.destroy()


def test_kimodo_2_outputs_and_details():
    _, path = _kimodo_clip((0.0, 0.0, 0.0))
    geo = hou.node("/obj").createNode("geo", "fxmotion_outputs")
    try:
        node = geo.createNode("vb::kimodo_motion::2.0")
        node.parm("clip_path").set(path.as_posix())
        counts = [len(node.geometry(i).points()) for i in range(4)]
        assert counts[0] > 1000, counts
        assert counts[1] == counts[2] == counts[3] == 77, counts
        g = node.geometry(2)
        assert g.attribValue("fxmotion_skeleton") == "soma77"
        assert g.attribValue("fxmotion_fps") == 30.0
        assert '"backend": "kimodo"' in g.attribValue("fxmotion_source")
        assert g.findPointAttrib("contact") is not None
        assert node.errors() == ()
        node.parm("last_error").set("boom")
        # cook() raises on a node in error, which is the point
        with contextlib.suppress(hou.OperationFailed):
            node.cook(force=True)
        assert any("boom" in e for e in node.errors())
    finally:
        geo.destroy()


def test_a_native_npz_is_a_clean_node_error():
    """Clip Path on a raw Kimodo NPZ (what 1.1 downloaded): the node goes red
    with the reason, not a Python traceback."""
    fixture = _repo() / "tests" / "fixtures" / "kimodo_stop.npz"
    geo = hou.node("/obj").createNode("geo", "fxmotion_native")
    try:
        node = geo.createNode("vb::kimodo_motion::2.0")
        node.parm("clip_path").set(fixture.as_posix())
        node.geometry(2)  # cooks output 2; its error stays on the inner SOP
        errors = " ".join(node.node("animated_pose").errors())
        assert "not an fxmotion.clip/1 clip" in errors, errors
        assert "Traceback" not in errors, errors
    finally:
        geo.destroy()


def _ardy_clip():
    import sys
    import tempfile
    from pathlib import Path

    import numpy as np

    from fxmotion import clipformat

    sys.path.insert(0, str(_repo() / "server"))
    import diffusion_adapter as da

    with np.load(_repo() / "tests" / "fixtures" / "ardy_path.npz") as z:
        npz = {k: z[k] for k in z.files}
    path = Path(tempfile.mkdtemp()) / "ardy.npz"
    clip = da.to_clip(npz, da.Canon(), da.ModelSpec("ardy", "ardy_core", 20.0))
    clipformat.save(path, clip)
    return clip, path


def test_ardy_1_outputs_and_details():
    import numpy as np

    from fxmotion import clip as fxclip

    clip, path = _ardy_clip()
    geo = hou.node("/obj").createNode("geo", "fxmotion_ardy")
    try:
        node = geo.createNode("vb::ardy_motion::1.0")
        node.parm("clip_path").set(path.as_posix())
        node.parm("retime").set(0)
        counts = [len(node.geometry(i).points()) for i in range(4)]
        assert counts == [9084, 27, 27, 27], counts
        hou.setFrame(31)  # start frame 1, no retime: sample 30
        g = node.geometry(2)
        assert g.attribValue("fxmotion_skeleton") == "ardy_core"
        assert g.attribValue("fxmotion_fps") == 20.0
        pos, _, _ = fxclip.joint_frames(clip, 30)
        got = np.array(g.pointFloatAttribValues("P")).reshape(-1, 3)
        assert np.abs(got - pos).max() < 1e-5
        assert node.parm("server_url").eval() == "http://localhost:8002"
        assert node.parm("seg_regen1") is None  # no Regenerate on ARDY
        assert node.errors() == ()
    finally:
        geo.destroy()


def test_ardy_pose_input_must_be_ardy_core():
    import json

    from fxmotion.nodes import ardy

    geo = hou.node("/obj").createNode("geo", "fxmotion_ardy_pose")
    try:
        node = geo.createNode("vb::ardy_motion::1.0")
        rig = geo.createNode("vb::kimodo_motion::2.0")  # a SOMA77 skeleton
        node.setInput(1, rig, 1)
        # a new node owns a timeline, so the key goes on its full-body track
        tl = json.loads(node.parm("timeline_json").eval())
        tl["tracks"] = {"fullbody": [1]}
        node.parm("timeline_json").set(json.dumps(tl))
        try:
            ardy.build_payload(node, 20.0)
        except ValueError as e:
            assert "ARDY_CORE" in str(e), str(e)
        else:
            raise AssertionError("a SOMA77 pose rig was accepted")
    finally:
        geo.destroy()


def run():
    results = []
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            try:
                fn()
                results.append("ok   " + name)
            except AssertionError as e:
                results.append("FAIL " + name + " -- " + str(e))
            except Exception as e:
                results.append(
                    "ERR  " + name + " -- %s: %s" % (type(e).__name__, e)
                )
    return results
