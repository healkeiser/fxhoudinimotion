"""Every tracked Python source is ASCII: non-ASCII characters in strings are
written as escapes, comments are ASCII (the repo rule, and the panel's
convention test extended to the whole repo)."""

import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_every_python_file_is_ascii():
    files = subprocess.run(
        ["git", "ls-files", "*.py"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    bad = []
    for name in files:
        if name.startswith(("vendor/", "houdini/otls/")):
            continue
        for n, line in enumerate(
            (REPO / name).read_bytes().splitlines(), start=1
        ):
            if any(b > 127 for b in line):
                bad.append("%s:%d" % (name, n))
    assert not bad, "non-ASCII source: %s" % ", ".join(bad)
