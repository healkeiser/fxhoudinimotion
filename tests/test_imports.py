"""Pure modules must import without Houdini: the model servers and CI have no
hou. Later tasks append to PURE as they add modules."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PURE = [
    "fxmotion",
    "fxmotion.skeletons",
    "fxmotion.clipformat",
    "kimodo_adapter",
    "fxmotion_server",
]


@pytest.mark.parametrize("module", PURE)
def test_imports_without_hou(module):
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO / "houdini" / "python"), str(REPO / "server")]
    )
    # sys.modules[name] = None makes `import name` raise ImportError
    code = "import sys; sys.modules['hou'] = None; import %s" % module
    subprocess.run([sys.executable, "-c", code], check=True, env=env)
