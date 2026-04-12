"""pytest configuration: ensure the repo root is on sys.path before any test
module is imported, so that grader, tasks, models etc. resolve without
triggering the relative-import path in the root __init__.py."""

import sys
import os

# Insert repo root at the front of sys.path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
