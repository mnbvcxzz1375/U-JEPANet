"""U-JEPANet: dual-path JEPA semi-replacement for 3D U-Net deep stages."""

from .model import build_model, UJEPAConfig

__all__ = ["build_model", "UJEPAConfig"]
__version__ = "0.1.0"
