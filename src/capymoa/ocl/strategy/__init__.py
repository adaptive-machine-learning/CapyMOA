"""Online Continual Learning (OCL) strategies."""

from . import l2p
from ._aser import ASER
from ._der import DER
from ._derpp import DERPP
from ._ewc import EWC
from ._experience_replay import ExperienceReplay
from ._gdumb import GDumb
from ._lpr import LPR
from ._lwf import LWF
from ._mas import MAS
from ._mir import MIR
from ._ncm import NCM
from ._pcr import PCR
from ._rar import RAR
from ._rwalk import RWalk
from ._scr import SCR
from ._si import SI
from ._slda import SLDA, RanPAC

__all__ = [
    "ASER",
    "DER",
    "DERPP",
    "EWC",
    "LPR",
    "LWF",
    "MAS",
    "MIR",
    "NCM",
    "PCR",
    "RAR",
    "SCR",
    "SI",
    "SLDA",
    "ExperienceReplay",
    "GDumb",
    "RWalk",
    "RanPAC",
    "l2p",
]
