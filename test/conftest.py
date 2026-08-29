import os
import sys

# Make the in-repo package (including tensorplay.testing) importable when
# running against a source checkout rather than an installed wheel.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
