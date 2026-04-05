"""
Ayase - Modular media quality metrics toolkit

"""

__version__ = "0.1.26"
__author__ = "Ayase Contributors"
__description__ = "Modular media quality metrics toolkit"

from .pipeline import AyasePipeline
from .profile import PipelineProfile, instantiate_profile_modules, load_profile

__all__ = [
    "__version__",
    "__author__",
    "__description__",
    "AyasePipeline",
    "PipelineProfile",
    "load_profile",
    "instantiate_profile_modules",
]
