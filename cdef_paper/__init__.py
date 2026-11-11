"""CDEF as printed: the paper version.

Every stage follows the CDEF paper, and the three methods it cites without
settings (BIMEF, CBM, IDE) follow their own papers. Nothing is taken from the
authors' released code except the one value the paper never gives (the
Gaussian of Eq. 1). README.md lists every gap and how it was filled.
"""
from .pipeline import enhance, load_config

__all__ = ["enhance", "load_config"]
