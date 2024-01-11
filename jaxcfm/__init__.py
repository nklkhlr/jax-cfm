from importlib.metadata import version

from .conditional_flow_matching import (
    CFM, ExactOptimalTransportCFM, TargetCFM, SchrodingerBridgeCFM)
from .optimal_transport import OTSampler


__version__ = version("jaxcfm")


__all__ = [
    "CFM",
    "ExactOptimalTransportCFM",
    "TargetCFM",
    "SchrodingerBridgeCFM",
    "OTSampler"
]
