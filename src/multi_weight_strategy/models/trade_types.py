from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Union

class TradeAction(Enum):
    ENTER = "ENTER"
    EXIT = "EXIT"

@dataclass
class Trade:
    timestamp: datetime
    action: TradeAction
    price: float
    position: int
    level: int = 0 