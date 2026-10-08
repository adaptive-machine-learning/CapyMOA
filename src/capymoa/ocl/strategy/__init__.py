"""Online Continual Learning (OCL) strategies."""

from . import l2p
from ._der import DER
from ._ewc import EWC
from ._experience_replay import ExperienceReplay
from ._gdumb import GDumb
from ._lwf import LWF
from ._mas import MAS
from ._ncm import NCM
from ._rar import RAR
from ._rwalk import RWalk
from ._si import SI
from ._slda import SLDA

__all__ = [
    "DER",
    "EWC",
    "LWF",
    "MAS",
    "NCM",
    "RAR",
    "SI",
    "SLDA",
    "ExperienceReplay",
    "GDumb",
    "RWalk",
    "l2p",
]
