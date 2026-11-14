"""Floor segmentation for Reachy Mini: which pixels of the head camera image are drivable floor.

Kept import-free on purpose: the renderer (floor_seg.thor_render) runs in its own Python 3.11 env without torch.
"""
import os

# Lens calibration of the Reachy Mini head camera (1280x720 mode) the synthetic data was rendered with.
DEFAULT_CALIB = os.path.join(os.path.dirname(__file__), "assets", "reachy_mini_camera.json")
