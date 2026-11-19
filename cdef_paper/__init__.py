"""CDEF: underwater image enhancement by colour discrimination and exposure fusion.

Every stage follows the CDEF paper, and the three methods it cites without
settings (BIMEF, CBM, IDE) are implemented from their own papers. The default
config (config.yaml) switches a few settings to the authors' released code;
paper_only.yaml keeps the papers' values. README.md lists every choice.
"""
from .pipeline import enhance, load_config

__all__ = ["enhance", "load_config"]
