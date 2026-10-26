from .face_analysis import *

def __getattr__(name):
    if name == "MaskRenderer":
        from .mask_renderer import MaskRenderer
        return MaskRenderer
    raise AttributeError(name)
