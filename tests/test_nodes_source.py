"""The HDA callbacks call names in fxmotion.nodes.kimodo and .common; a
rename that forgets the asset breaks every button silently. Checked from
source, without hou."""

import ast
from pathlib import Path

NODES = (
    Path(__file__).resolve().parents[1]
    / "houdini"
    / "python"
    / "fxmotion"
    / "nodes"
)

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
    "common.py": {
        "cook_animated",
        "cook_rest",
        "cook_file",
        "load_skeleton_file",
    },
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
