"""fxmotion: NVIDIA motion models (Kimodo, ARDY, MotionBricks) in Houdini.

Importable without Houdini. The model servers import fxmotion.clipformat and
fxmotion.skeletons, and the offline tests import every module that does not
need hou. Modules that need Houdini live under fxmotion.nodes and
fxmotion.timeline, or import hou inside their functions.
"""
