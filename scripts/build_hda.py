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
    return "from fxmotion.nodes import kimodo; kimodo.%s(kwargs['node']%s)" % (
        fn,
        extra,
    )


def button(name, label, fn, *args, **kw):
    return hou.ButtonParmTemplate(
        name,
        label,
        script_callback=cb(fn, *args),
        script_callback_language=PY,
        **kw,
    )


def hidden_string(name, label, **kw):
    return hou.StringParmTemplate(
        name, label, 1, default_value=("",), is_hidden=True, **kw
    )


def parms():
    ptg = hou.ParmTemplateGroup()

    gen = hou.FolderParmTemplate(
        "fld_generate", "Generate", folder_type=hou.folderType.Tabs
    )
    gen.addParmTemplate(
        button(
            "open_timeline",
            "Open Timeline",
            "open_timeline",
            is_label_hidden=True,
            join_with_next=True,
            help="Open the __Motion Timeline__ panel for this node: prompt "
            "segments, transitions and pose tracks.",
        )
    )
    gen.addParmTemplate(
        button(
            "generate",
            "Generate",
            "generate",
            is_label_hidden=True,
            join_with_next=True,
            help="Send the request to the server. Progress shows in "
            "__Status__ and under the node.",
        )
    )
    gen.addParmTemplate(
        button(
            "cancel",
            "Cancel",
            "cancel",
            is_label_hidden=True,
            help="Cancel the queued job or discard the running one.",
        )
    )
    gen.addParmTemplate(
        hou.MenuParmTemplate(
            "model",
            "Model",
            MODELS,
            default_value=0,
            help="__RP__ = Bones Rigplay 1 (~700 h of mocap), the recommended "
            "default. __SEED__ = BONES-SEED (288 h, public data).",
        )
    )
    gen.addParmTemplate(
        hou.ToggleParmTemplate(
            "force",
            "Force Regenerate",
            default_value=True,
            help="Run inference again even if the server has an identical "
            "request cached.",
        )
    )
    gen.addParmTemplate(
        hou.IntParmTemplate(
            "seed",
            "Seed",
            1,
            default_value=(-1,),
            min=-1,
            max=100000,
            min_is_strict=True,
            help="`-1` = a new random result each time; any other value "
            "repeats a result.",
        )
    )
    seg = hou.FolderParmTemplate(
        "segments",
        "Segments",
        folder_type=hou.folderType.ScrollingMultiparmBlock,
    )
    seg.setDefaultValue(1)
    sync = cb("sync_segments")
    seg.addParmTemplate(
        hou.StringParmTemplate(
            "seg_prompt#",
            "Prompt",
            1,
            default_value=("",),
            script_callback=sync,
            script_callback_language=PY,
            help=PROMPT_HELP,
        )
    )
    seg.addParmTemplate(
        hou.IntParmTemplate(
            "seg_from#", "Frames", 1, default_value=(0,), is_hidden=True
        )
    )
    seg.addParmTemplate(
        hou.IntParmTemplate(
            "seg_to#", "to", 1, default_value=(0,), is_hidden=True
        )
    )
    seg.addParmTemplate(
        hou.LabelParmTemplate(
            "seg_range#",
            "Frames",
            join_with_next=True,
            column_labels=(
                '`chs("seg_from#")` - `chs("seg_to#")`   '
                '(`rint(ch("seg_frames#") / ch("scene_fps") * 100) / 100` s)',
            )
            + ("",) * 15,
        )
    )
    seg.addParmTemplate(
        hou.IntParmTemplate(
            "seg_frames#",
            "Length",
            1,
            default_value=(48,),
            min=1,
            max=240,
            min_is_strict=True,
            join_with_next=True,
            script_callback=sync,
            script_callback_language=PY,
            help="Length of this segment in scene frames.",
        )
    )
    seg.addParmTemplate(
        button(
            "seg_split#",
            "Split",
            "split_segment",
            INDEX,
            join_with_next=True,
            help="Cut this segment in two at the playhead.",
        )
    )
    seg.addParmTemplate(
        button(
            "seg_regen#",
            "Regenerate",
            "regenerate",
            INDEX,
            "False",
            join_with_next=True,
            help="Re-roll this segment alone; the clip keeps its length and "
            "both joins stay continuous.",
        )
    )
    seg.addParmTemplate(
        button(
            "seg_regen_end#",
            "From Here",
            "regenerate",
            INDEX,
            "True",
            help="Re-roll this segment and every segment after it.",
        )
    )
    gen.addParmTemplate(seg)
    gen.addParmTemplate(
        hou.StringParmTemplate(
            "prompt",
            "Prompt",
            1,
            default_value=("a person walks forward",),
            is_hidden=True,
        )
    )
    gen.addParmTemplate(
        hou.IntParmTemplate(
            "duration_frames",
            "Duration (frames)",
            1,
            default_value=(72,),
            is_hidden=True,
        )
    )
    gen.addParmTemplate(hidden_string("status", "Status"))
    gen.addParmTemplate(hidden_string("clip_info", "Clip"))
    gen.addParmTemplate(
        hou.FloatParmTemplate(
            "scene_fps",
            "Scene FPS",
            1,
            default_expression=("$FPS",),
            default_expression_language=(hou.scriptLanguage.Hscript,),
            is_hidden=True,
        )
    )
    ptg.append(gen)

    con = hou.FolderParmTemplate(
        "fld_constraints", "Constraints", folder_type=hou.folderType.Tabs
    )
    path = hou.FolderParmTemplate(
        "grp_path",
        "Root Path (input 0)",
        folder_type=hou.folderType.Collapsible,
        tags={"group_default": "1"},
    )
    path.addParmTemplate(
        hou.IntParmTemplate(
            "path_waypoints",
            "Path Waypoints",
            1,
            default_value=(8,),
            min=0,
            max=64,
            min_is_strict=True,
            help="The curve is thinned to this many points by arc length. "
            "`0` = every point.",
        )
    )
    con.addParmTemplate(path)
    js = hou.FolderParmTemplate(
        "grp_json",
        "Constraints JSON",
        folder_type=hou.folderType.Collapsible,
        tags={"group_default": "0"},
    )
    js.addParmTemplate(
        hou.StringParmTemplate(
            "constraints_file",
            "Constraints File",
            1,
            default_value=("",),
            string_type=hou.stringParmType.FileReference,
            file_type=hou.fileType.Any,
            tags={"filechooser_pattern": "*.json"},
            help="Kimodo constraints JSON, in Kimodo's own model space. "
            "Ignored when Constraints JSON is set.",
        )
    )
    js.addParmTemplate(
        hou.StringParmTemplate(
            "constraints_json",
            "Constraints JSON",
            1,
            default_value=("",),
            tags={"editor": "1", "editorlines": "3-8"},
            help="Inline Kimodo constraints JSON (a list of constraint "
            "dicts), in Kimodo's model space.",
        )
    )
    con.addParmTemplate(js)
    pose = hou.FolderParmTemplate(
        "grp_pose",
        "Pose Keyframes (input 1)",
        folder_type=hou.folderType.Collapsible,
        tags={"group_default": "1"},
    )
    pose.addParmTemplate(
        button(
            "make_pose_rig",
            "Create Pose Rig",
            "make_pose_rig",
            help="Drop a capture-pose rig wired to input 1, ready to pose.",
        )
    )
    pose.addParmTemplate(
        hou.StringParmTemplate(
            "pose_keyframes",
            "Pose Keyframes",
            1,
            default_value=("",),
            disable_when=TIMELINE_OWNS,
            help="Scene frames to sample input 1 at, e.g. `1 45 89`. "
            "Disabled while the timeline owns the keys.",
        )
    )
    pose.addParmTemplate(
        hou.MenuParmTemplate(
            "pose_type",
            "Pose Constraint",
            ("Full-Body", "End-Effector"),
            default_value=0,
            disable_when='{ pose_keyframes == "" } ' + TIMELINE_OWNS,
        )
    )
    ee = (
        '{ pose_type != "End-Effector" } { pose_keyframes == "" } '
        + TIMELINE_OWNS
    )
    for name, label, on, join in (
        ("ee_left_hand", "Left Hand", False, True),
        ("ee_right_hand", "Right Hand", True, False),
        ("ee_left_foot", "Left Foot", False, True),
        ("ee_right_foot", "Right Foot", False, False),
    ):
        pose.addParmTemplate(
            hou.ToggleParmTemplate(
                name,
                label,
                default_value=on,
                disable_when=ee,
                join_with_next=join,
            )
        )
    con.addParmTemplate(pose)
    ptg.append(con)

    out = hou.FolderParmTemplate(
        "fld_output", "Output", folder_type=hou.folderType.Tabs
    )
    out.addParmTemplate(
        hou.IntParmTemplate(
            "start_frame",
            "Start Frame",
            1,
            script_callback=cb("refresh_starts"),
            script_callback_language=PY,
            default_expression=("$FSTART",),
            default_expression_language=(hou.scriptLanguage.Hscript,),
            min=-1000,
            max=1000,
            help="Scene frame on which the clip begins.",
        )
    )
    out.addParmTemplate(
        hou.StringParmTemplate(
            "clip_path",
            "Clip Path",
            1,
            default_value=("",),
            string_type=hou.stringParmType.FileReference,
            file_type=hou.fileType.Any,
            tags={"filechooser_pattern": "*.npz"},
            help="The fxmotion clip the node reads. Set by Generate; any "
            "fxmotion.clip/1 file works without a server.",
        )
    )
    adv = hou.FolderParmTemplate(
        "grp_advanced",
        "Advanced",
        folder_type=hou.folderType.Collapsible,
        tags={"group_default": "0"},
    )
    adv.addParmTemplate(
        hou.ToggleParmTemplate(
            "retime",
            "Retime to Scene FPS",
            default_value=True,
            help="Keep the clip's real duration at any `$FPS` (nearest "
            "sample). Off = one clip sample per scene frame.",
        )
    )
    out.addParmTemplate(adv)
    ptg.append(out)

    srv = hou.FolderParmTemplate(
        "fld_server", "Server", folder_type=hou.folderType.Tabs
    )
    srv.addParmTemplate(
        hou.StringParmTemplate(
            "server_url",
            "API Server URL",
            1,
            default_value=("http://localhost:8001",),
            join_with_next=True,
        )
    )
    srv.addParmTemplate(
        button("test_connection", "Test Connection", "test_connection")
    )
    srv.addParmTemplate(
        hou.StringParmTemplate(
            "download_dir",
            "Download Dir",
            1,
            default_value=("$HIP/fxmotion_cache",),
            string_type=hou.stringParmType.FileReference,
            file_type=hou.fileType.Directory,
        )
    )
    ptg.append(srv)

    for name, label in (("job_id", "Job ID"), ("last_error", "Last Error")):
        ptg.append(hidden_string(name, label))
    ptg.append(
        hou.FloatParmTemplate(
            "progress",
            "Progress",
            1,
            default_value=(0.0,),
            min=0.0,
            max=1.0,
            is_hidden=True,
        )
    )
    ptg.append(hidden_string("timeline_json", "Timeline", tags={"editor": "1"}))
    ptg.append(
        hou.ToggleParmTemplate(
            "has_timeline", "Has Timeline", default_value=False, is_hidden=True
        )
    )
    ptg.append(
        hou.IntParmTemplate(
            "frame_ref",
            "Frame",
            1,
            default_expression=("$F",),
            default_expression_language=(hou.scriptLanguage.Hscript,),
            is_hidden=True,
        )
    )
    return ptg


def _patch_dialog_script(definition, labels):
    """Input/output connector labels and the Segments minimum live only in
    the DialogScript."""
    ds = definition.sections()["DialogScript"].contents()
    ds = ds.splitlines(keepends=True)
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
    after = max(
        i for i, line in enumerate(ds) if line.lstrip().startswith("inputlabel")
    )
    inject = "".join(
        '    outputlabel\t%d\t"%s"\n' % (i + 1, lbl)
        for i, lbl in enumerate(labels)
    )
    definition.addSection(
        "DialogScript",
        "".join(ds[: after + 1]) + inject + "".join(ds[after + 1 :]),
    )


def build():
    geo = hou.node("/obj").createNode("geo", "kimodo_motion_build")
    try:
        subnet = geo.createNode("subnet", "kimodo_motion")
        cooks = (
            (
                "rest_geometry",
                "common.cook_file(hou.pwd(), %r, 'skin')" % SKELETON,
            ),
            (
                "capture_pose",
                "common.cook_file(hou.pwd(), %r, 'capture_pose')" % SKELETON,
            ),
            ("animated_pose", "common.cook_animated(hou.pwd())"),
            ("t_pose", "common.cook_rest(hou.pwd(), %r)" % SKELETON),
        )
        # output connector colours, as kinefx::characterio::2.0
        colors = {0: (0.584, 0.776, 1.0), 2: (0.976, 0.780, 0.263)}
        for idx, (name, call) in enumerate(cooks):
            sop = subnet.createNode("python", name)
            sop.parm("python").set(
                "from fxmotion.nodes import common\n%s\n" % call
            )
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
            name=NAME,
            hda_file_name=str(packed),
            description=LABEL,
            min_num_inputs=0,
            max_num_inputs=2,
            version=VERSION,
        )
        d = node.type().definition()
        d.setMaxNumOutputs(len(cooks))
        d.setIcon("kimodo_motion.svg")
        d.setParmTemplateGroup(parms())
        d.addSection(
            "OnCreated",
            "from fxmotion.nodes import kimodo\n"
            "kimodo.on_created(kwargs['node'])\n",
        )
        d.setExtraFileOption("OnCreated/IsPython", True)
        d.addSection("DescriptiveParmName", "status")
        d.addSection("Help", HELP)
        shelf = d.sections().get("Tools.shelf")
        if shelf is not None:
            d.addSection(
                "Tools.shelf",
                re.sub(
                    r"<toolSubmenu>.*?</toolSubmenu>",
                    "<toolSubmenu>fxmotion</toolSubmenu>",
                    shelf.contents(),
                    count=1,
                ),
            )
        d.save(str(packed))
        _patch_dialog_script(
            d, ["Rest Geometry", "Capture Pose", "Animated Pose", "T-Pose"]
        )
        d.save(str(packed))
        if LIBRARY.exists():
            shutil.rmtree(LIBRARY)
        hou.hda.expandToDirectory(str(packed), str(LIBRARY))
        print("HDA saved: %s  type: %s" % (LIBRARY, NAME))
    finally:
        geo.destroy()


build()
