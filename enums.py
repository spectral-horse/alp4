from enum import Enum
from ._consts import *



class AlpTrigger(Enum):
    NONE = ALP_DEFAULT
    LOW = ALP_LEVEL_LOW
    HIGH = ALP_LEVEL_HIGH
    RISING = ALP_EDGE_RISING
    FALLING = ALP_EDGE_FALLING

class AlpDataFormat(Enum):
    MSB_ALIGN = ALP_DATA_MSB_ALIGN
    LSB_ALIGN = ALP_DATA_LSB_ALIGN
    BINARY_TOPDOWN = ALP_DATA_BINARY_TOPDOWN
    BINARY_BOTTOMUP = ALP_DATA_BINARY_BOTTOMUP
