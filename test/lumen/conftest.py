"""Shared pytest configuration for the lumen tests.

Matplotlib is switched to the non-interactive ``Agg`` backend before any
visualization module is imported, so tests never pop up plot windows.
"""

import matplotlib

matplotlib.use("Agg")