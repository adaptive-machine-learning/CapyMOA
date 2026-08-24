"""Online Continual Learning (OCL) strategies."""

from ._experience_replay import ExperienceReplay
from ._slda import SLDA
from ._ncm import NCM
from ._gdumb import GDumb
from ._rar import RAR
from . import l2p
from ._ewc import EWC
from ._si import SI

__all__ = ["ExperienceReplay", "SLDA", "NCM", "GDumb", "RAR", "l2p", "EWC", "SI"]
