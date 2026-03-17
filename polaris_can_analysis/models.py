from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ParsedFrame:
    timestamp: str
    elapsed_s: float
    interface: str
    can_id: str
    can_id_int: int
    frame_name: str
    dlc: int
    data: List[int]
    raw_message: str
    parse_warning: str = ""


@dataclass
class OnWaterDetection:
    start_s: Optional[float]
    steady_state_us_cm: Optional[float]
    threshold_us_cm: Optional[float]
