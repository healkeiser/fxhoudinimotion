"""The Motion Timeline panel for fxmotion generator nodes.

model.py   timeline arithmetic (pure python, tested)
bridge.py  reading/writing the node (hou)
widget.py  the Qt panel (binding chosen by qt.py)
"""


def create_widget():
    """Entry point used by houdini/python_panels/fxmotion_timeline.pypanel."""
    from .widget import TimelineWidget

    return TimelineWidget()
