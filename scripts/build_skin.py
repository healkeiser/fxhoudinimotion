"""Build a skeleton's rest geometry (skin + bind skeleton), run with hython:

    hython scripts/build_skin.py <skeleton> [skin_standard.npz] [out_dir]

Produces two .bgeo.sc files, read by the fxmotion skeleton registry:
  - skin.bgeo.sc  : the body mesh in its A-pose bind, with a KineFX
                    `boneCapture` attribute (weights from Kimodo LBS, bind from
                    bind_rig_transform). Drive it with kinefx::jointdeform
                    (in0=this, in1=A-pose rest skeleton, in2=animated skeleton).
  - apose.bgeo.sc : the A-pose rest skeleton (feet on floor) that the
                    mesh is bound to.

No torch needed: skin_standard.npz is a plain .npz, and the T-pose offsets (tp)
come from the registry's rest rotations. The bind orientation uses bind_rig @
tp so the skeletons stay bone-aligned and consistent with the animated output0
(the per-joint tp offset cancels in jointdeform's anim @ inv(bind)).
"""

import sys
from pathlib import Path

import hou
import numpy as np

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "houdini" / "python")
)
from fxmotion import skeletons  # noqa: E402

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
# where each skeleton's skin_standard.npz lives when the repos sit side by side
_DEFAULT_NPZ = {
    "soma77": _REPO.parent
    / "kimodo"
    / "kimodo"
    / "assets"
    / "skeletons"
    / "somaskel77",
    "ardy_core": _REPO.parent
    / "ardy"
    / "ardy"
    / "assets"
    / "skeletons"
    / "cskel27",
}
_SKELETON = sys.argv[1] if len(sys.argv) > 1 else "soma77"
_SKIN_NPZ = (
    Path(sys.argv[2])
    if len(sys.argv) > 2
    else _DEFAULT_NPZ[_SKELETON] / "skin_standard.npz"
)
_OUT_DIR = (
    Path(sys.argv[3])
    if len(sys.argv) > 3
    else _REPO / "houdini" / "python" / "fxmotion" / "skeletons" / "data"
)


def _tpose_offsets():
    """Column-vector offsets that make each joint frame bone-aligned: the
    transpose of the registry's row-vector rest rotations (identity for
    skeletons whose frames are world-aligned)."""
    rest = skeletons.get(_SKELETON).rest_rot
    return np.swapaxes(np.asarray(rest, dtype=np.float64), -1, -2)


def _houdini_world_rot(col_vec_rot):
    """Kimodo column-vector rotation -> Houdini row-major float[9]."""
    return tuple(float(v) for v in col_vec_rot.T.flatten())


def _load():
    s = np.load(_SKIN_NPZ, allow_pickle=True)
    d = {
        "bv": s["bind_vertices"].astype(np.float64),
        "faces": s["faces"].astype(np.int64),
        "idx": s["lbs_indices"].astype(np.int64),
        "w": s["lbs_weights"].astype(np.float64),
        "brt": s["bind_rig_transform"].astype(np.float64),
        "names": [str(x) for x in s["rig_joint_names"]],
        "conn": s["rig_joint_connections"].astype(np.int64),
    }
    want = list(skeletons.get(_SKELETON).joint_names)
    if d["names"] != want:
        raise ValueError(
            "%s skin joints differ from the registry's order" % _SKELETON
        )
    return d


def build_skin_geo(d):
    """Body mesh + capture, packed into a `boneCapture` index-pair attrib."""
    tp = _tpose_offsets()
    bv, faces, idx, w, brt = d["bv"], d["faces"], d["idx"], d["w"], d["brt"]
    names = d["names"]
    J, V = len(names), bv.shape[0]

    # World bind matrix per joint (Houdini row-vector / row-major): rot =
    # (bind@tp).T, translation in the last row. pCaptData = its inverse (world
    # -> rest) + 4 tube params.
    pcapt = np.zeros((J, 20))
    for j in range(J):
        Wb = np.identity(4)
        Wb[:3, :3] = (brt[j, :3, :3] @ tp[j]).T
        Wb[3, :3] = brt[j, :3, 3]
        pcapt[j, :16] = np.linalg.inv(Wb).flatten()
        pcapt[j, 16:] = 1.0

    g = hou.Geometry()
    pts = [g.createPoint() for _ in range(V)]
    for i, p in enumerate(pts):
        p.setPosition(hou.Vector3(bv[i].tolist()))
    # Reverse winding: Kimodo's faces are wound opposite to Houdini's
    # outward-normal convention, so flip each face's vertex order.
    g.createPolygons(tuple(tuple(int(x) for x in f[::-1]) for f in faces))

    g.addArrayAttrib(
        hou.attribType.Point, "boneCapture_index", hou.attribData.Int
    )
    g.addArrayAttrib(
        hou.attribType.Point, "boneCapture_data", hou.attribData.Float
    )
    for i, p in enumerate(pts):
        m = w[i] > 1e-6
        p.setAttribValue("boneCapture_index", [int(x) for x in idx[i][m]])
        p.setAttribValue("boneCapture_data", [float(x) for x in w[i][m]])
    g.addArrayAttrib(
        hou.attribType.Global, "boneCapture_pCaptPath", hou.attribData.String
    )
    g.setGlobalAttribValue("boneCapture_pCaptPath", names)
    g.addArrayAttrib(
        hou.attribType.Global,
        "boneCapture_pCaptData",
        hou.attribData.Float,
        tuple_size=20,
    )
    g.setGlobalAttribValue(
        "boneCapture_pCaptData", [float(x) for x in pcapt.flatten()]
    )

    # Pack the unpacked attribs into the `boneCapture` index-pair attribute.
    obj = hou.node("/obj")
    container = obj.createNode("geo", "_skinpack")
    container.deleteItems(container.children())
    tmp = _OUT_DIR / "_skin_unpacked.bgeo.sc"
    g.saveToFile(str(tmp))
    fsop = container.createNode("file")
    fsop.parm("file").set(str(tmp))
    pack = container.createNode("captureattribpack")
    pack.setInput(0, fsop)
    packed = pack.geometry().freeze()
    container.destroy()
    tmp.unlink()
    return packed


def build_apose_skeleton(d):
    """A-pose rest skeleton (feet on floor): name, transform, parent lines."""
    tp = _tpose_offsets()
    brt, names, conn = d["brt"], d["names"], d["conn"]
    J = len(names)
    g = hou.Geometry()
    g.addAttrib(hou.attribType.Point, "name", "")
    g.addAttrib(
        hou.attribType.Point, "transform", (1.0, 0, 0, 0, 1, 0, 0, 0, 1)
    )
    pts = []
    for j in range(J):
        p = g.createPoint()
        p.setPosition(hou.Vector3(brt[j, :3, 3].tolist()))
        p.setAttribValue("name", names[j])
        p.setAttribValue(
            "transform", _houdini_world_rot(brt[j, :3, :3] @ tp[j])
        )
        pts.append(p)
    for a, b in conn:
        poly = g.createPolygon()
        poly.setIsClosed(False)
        poly.addVertex(pts[int(a)])
        poly.addVertex(pts[int(b)])
    return g


def main():
    if not _SKIN_NPZ.exists():
        raise FileNotFoundError(
            f"skin_standard.npz not found at {str(_SKIN_NPZ)!r}. Pass it explicitly: "
            "hython scripts/build_skin.py <skeleton> /path/to/skin_standard.npz [out_dir]"
        )
    d = _load()
    skin = build_skin_geo(d)
    apose = build_apose_skeleton(d)
    skin_path = _OUT_DIR / ("%s_skin.bgeo.sc" % _SKELETON)
    apose_path = _OUT_DIR / ("%s_apose.bgeo.sc" % _SKELETON)
    skin.saveToFile(str(skin_path))
    apose.saveToFile(str(apose_path))
    bc = skin.findPointAttrib("boneCapture")
    print(
        f"skin:  {len(skin.points())} pts / {len(skin.prims())} prims / boneCapture size "
        f"{bc.size() if bc else None} -> {skin_path}"
    )
    print(
        f"apose: {len(apose.points())} pts / {len(apose.prims())} prims -> {apose_path}"
    )


if __name__ == "__main__":
    main()
