from enum import Enum
from ._consts import *



class AlpTrigger(Enum):
    NONE = ALP_DEFAULT
    LOW = ALP_LEVEL_LOW
    HIGH = ALP_LEVEL_HIGH
    RISING = ALP_EDGE_RISING
    FALLING = ALP_EDGE_FALLING
