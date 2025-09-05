from ctypes import CDLL, c_long, c_void_p, byref
from enum import Enum
import numpy as np



class AlpError(RuntimeError):
    _err_codes = {
        1001: "NOT_ONLINE",
        1002: "NOT_IDLE",
        1003: "NOT_AVAILABLE",
        1004: "NOT_READY",
        1005: "PARM_INVALID",
        1006: "ADDR_INVALID",
        1007: "MEMORY_FULL",
        1008: "SEQ_IN_USE",
        1009: "HALTED",
        1010: "ERROR_INIT",
        1011: "ERROR_COMM",
        1012: "DEVICE_REMOVED",
        1013: "NOT_CONFIGURED",
        1014: "LOADER_VERSION",
        1018: "ERROR_POWER_DOWN",
        1019: "DRIVER_VERSION",
        1020: "SDRAM_INIT"
    }

    def __init__(self, code):
        if code == _ALP_OK.value:
            raise ValueError("code 0 is ALP_OK, not an error")

        super().__init__(self._err_codes.get(code, "UNKNOWN"))


_ALP_DEFAULT = c_long(0)
_ALP_ENABLE = c_long(1)
_ALP_OK = c_long(0)
_ALP_DEVICE_NUMBER = c_long(2000)
_ALP_VERSION = c_long(2001)
_ALP_AVAIL_MEMORY = c_long(2003)
_ALP_SYNCH_POLARITY = c_long(2004)
_ALP_LEVEL_HIGH = c_long(2006)
_ALP_LEVEL_LOW = c_long(2007)
_ALP_TRIGGER_EDGE = c_long(2005)
_ALP_EDGE_FALLING = c_long(2008)
_ALP_EDGE_RISING = c_long(2009)
_ALP_DEV_DMDTYPE = c_long(2021)
_ALP_DMDTYPE_XGA = c_long(1)
_ALP_DMDTYPE_1080P_095A = c_long(3)
_ALP_DMDTYPE_XGA_07A = c_long(4)
_ALP_DMDTYPE_XGA_055X = c_long(6)
_ALP_DMDTYPE_WUXGA_096A = c_long(7)
_ALP_DMDTYPE_WQXGA_400MHZ_090A = c_long(8)
_ALP_DMDTYPE_WQXGA_480MHZ_090A = c_long(9)
_ALP_DMDTYPE_1080P_065A = c_long(10)
_ALP_DMDTYPE_1080P_065_S600 = c_long(11)
_ALP_DMDTYPE_WXGA_S450 = c_long(12)
_ALP_DMDTYPE_DLPC910REV = c_long(254)
_ALP_DMDTYPE_DISCONNECT = c_long(255)
_ALP_USB_CONNECTION = c_long(2016)
_ALP_DEV_DYN_SYNCH_OUT1_GATE = c_long(2023)
_ALP_DEV_DYN_SYNCH_OUT2_GATE = c_long(2024)
_ALP_DEV_DYN_SYNCH_OUT3_GATE = c_long(2025)
_ALP_DDC_FPGA_TEMPERATURE = c_long(2050)
_ALP_APPS_FPGA_TEMPERATURE = c_long(2051)
_ALP_PCB_TEMPERATURE = c_long(2052)
_ALP_DEV_DISPLAY_HEIGHT = c_long(2057)
_ALP_DEV_DISPLAY_WIDTH = c_long(2058)
_ALP_SEQ_DMD_LINES = c_long(2125)
_ALP_PWM_LEVEL = c_long(2063)
_ALP_DEV_DMD_MODE = c_long(2064)
_ALP_DMD_RESUME = c_long(0)
_ALP_DMD_POWER_FLOAT = c_long(1)
_ALP_USB_DISCONNECT_BEHAVIOUR = c_long(2078)
_ALP_USB_IGNORE = c_long(1)
_ALP_USB_RESET = c_long(2)
_ALP_BITPLANES = c_long(2200)
_ALP_BITNUM = c_long(2103)
_ALP_BIN_MODE = c_long(2104)
_ALP_BIN_NORMAL = c_long(2105)
_ALP_BIN_UNINTERRUPTED = c_long(2106)
_ALP_PICNUM = c_long(2201)
_ALP_FIRSTFRAME = c_long(2101)
_ALP_LASTFRAME = c_long(2102)
_ALP_FIRSTLINE = c_long(2111)
_ALP_LASTLINE = c_long(2112)
_ALP_LINE_INC = c_long(2113)
_ALP_SCROLL_FROM_ROW = c_long(2123)
_ALP_SCROLL_TO_ROW = c_long(2124)
_ALP_SEQ_REPEAT = c_long(2100)
_ALP_PICTURE_TIME = c_long(2203)
_ALP_MIN_PICTURE_TIME = c_long(2211)
_ALP_MAX_PICTURE_TIME = c_long(2213)
_ALP_ILLUMINATE_TIME = c_long(2204)
_ALP_MIN_ILLUMINATE_TIME = c_long(2212)
_ALP_ON_TIME = c_long(2214)
_ALP_OFF_TIME = c_long(2215)
_ALP_SYNCH_DELAY = c_long(2205)
_ALP_MAX_SYNCH_DELAY = c_long(2209)
_ALP_SYNCH_PULSEWIDTH = c_long(2206)
_ALP_TRIGGER_IN_DELAY = c_long(2207)
_ALP_MAX_TRIGGER_IN_DELAY = c_long(2210)
_ALP_DATA_FORMAT = c_long(2110)
_ALP_DATA_MSB_ALIGN = c_long(0)
_ALP_DATA_LSB_ALIGN = c_long(1)
_ALP_DATA_BINARY_TOPDOWN = c_long(2)
_ALP_DATA_BINARY_BOTTOMUP = c_long(3)
_ALP_SEQ_PUT_LOCK = c_long(2119)
_ALP_FLUT_MODE = c_long(2118)
_ALP_FLUT_NONE = c_long(0)
_ALP_FLUT_9BIT = c_long(1)
_ALP_FLUT_18BIT = c_long(2)
_ALP_FLUT_ENTRIES9 = c_long(2120)
_ALP_FLUT_OFFSET9 = c_long(2122)
_ALP_PWM_MODE = c_long(2107)
_ALP_FLEX_PWM = c_long(3)
_ALP_DMD_MASK_SELECT = c_long(2134)
_ALP_DMD_MASK_16X16 = c_long(1)
_ALP_DMD_MASK_16X8 = c_long(2)
_ALP_SEQ_DMD_LINES = c_long(2125)
_ALP_PROJ_MODE = c_long(2300)
_ALP_MASTER = c_long(2301)
_ALP_SLAVE = c_long(2302)
_ALP_PROJ_STEP = c_long(2329)
_ALP_PROJ_STATE = c_long(2400)
_ALP_PROJ_ACTIVE = c_long(1200)
_ALP_PROJ_IDLE = c_long(1201)
_ALP_PROJ_INVERSION = c_long(2306)
_ALP_PROJ_UPSIDE_DOWN = c_long(2307)
_ALP_PROJ_QUEUE_MODE = c_long(2314)
_ALP_PROJ_LEGACY = c_long(0)
_ALP_PROJ_SEQUENCE_QUEUE = c_long(1)
_ALP_PROJ_QUEUE_ID = c_long(2315)
_ALP_PROJ_QUEUE_MAX_AVAIL = c_long(2316)
_ALP_PROJ_QUEUE_AVAIL = c_long(2317)
_ALP_PROJ_PROGRESS = c_long(2318)
_ALP_FLAG_QUEUE_IDLE = c_long(1)
_ALP_FLAG_SEQUENCE_ABORTING = c_long(2)
_ALP_FLAG_SEQUENCE_INDEFINITE = c_long(4)
_ALP_FLAG_FRAME_FINISHED = c_long(8)
_ALP_PROJ_RESET_QUEUE = c_long(2319)
_ALP_PROJ_ABORT_SEQUENCE = c_long(2320)
_ALP_PROJ_ABORT_FRAME = c_long(2321)
_ALP_PROJ_WAIT_UNTIL = c_long(2323)
_ALP_PROJ_WAIT_PIC_TIME = c_long(0)
_ALP_PROJ_WAIT_ILLU_TIME = c_long(1)
_ALP_FLUT_MAX_ENTRIES9 = c_long(2324)
_ALP_FLUT_WRITE_9BIT = c_long(2325)
_ALP_FLUT_WRITE_18BIT = c_long(2326)
_ALP_DMD_MASK_WRITE = c_long(2339)
_ALP_PUT_LINES = c_long(1)



class Alp:
    _lib = None

    def __init__(self, library_path):
        if self._lib is not None:
            raise RuntimeError("ALP already initialised")

        type(self)._lib = CDLL(library_path)

    def _call(self, func_name, *args):
        ret = self._lib[func_name](*args)

        if ret != _ALP_OK.value: raise AlpError(ret)

    def open_device(self, device_num = _ALP_DEFAULT.value):
        dev = AlpDevice()
        dev._id = c_long(0)
        dev._alp = self

        self._call(
            "AlpDevAlloc",
            c_long(device_num), c_long(0), byref(dev._id)
        )

        return dev


class AlpTrigger(Enum):
    NONE = _ALP_DEFAULT
    LOW = _ALP_LEVEL_LOW
    HIGH = _ALP_LEVEL_HIGH
    RISING = _ALP_EDGE_RISING
    FALLING = _ALP_EDGE_FALLING


class AlpDevice:
    def _inquire(self, inquiry, proj = False):
        func_name = "AlpProjInquire" if proj else "AlpDevInquire"
        value = c_long(0)

        self._alp._call(func_name, self._id, inquiry, byref(value))

        return value

    def _control(self, param, value):
        self._alp._call(
            "AlpProjControl",
            self._id, param, value
        )

    def get_display_size(self):
        width = self._inquire(_ALP_DEV_DISPLAY_WIDTH).value
        height = self._inquire(_ALP_DEV_DISPLAY_HEIGHT).value

        return width, height

    def allocate_sequence(self, bitplanes, images):
        seq = AlpSequence()
        seq._alp = self._alp
        seq._id = c_long(0)
        seq._dev = self

        self._alp._call(
            "AlpSeqAlloc",
            self._id, c_long(bitplanes), c_long(images), byref(seq._id)
        )

        return seq

    def halt(self):
        self._alp._call("AlpDevHalt", self._id)

    def wait(self):
        self._alp._call("AlpProjWait", self._id)

    def is_projecting(self):
        code = self._inquire(_ALP_PROJ_STATE, proj = True)

        if code.value == _ALP_PROJ_ACTIVE.value: return True
        elif code.value == _ALP_PROJ_IDLE.value: return False
        else:
            raise RuntimeError("AlpProjInquire gave invalid ALP_PROJ_STATE")

    def set_trigger(self, trigger_type):
        self._control(_ALP_PROJ_STEP, trigger_type.value)


class AlpSequence:
    _formats = {
        "msb_align": _ALP_DATA_MSB_ALIGN,
        "lsb_align": _ALP_DATA_LSB_ALIGN,
        "binary_topdown": _ALP_DATA_BINARY_TOPDOWN,
        "binary_bottomup": _ALP_DATA_BINARY_BOTTOMUP
    }

    def _control(self, param, value):
        self._alp._call(
            "AlpSeqControl",
            self._dev._id, self._id, param, value
        )

    def free(self):
        self._alp._call("AlpSeqFree", self._dev._id, self._id)

    def set_timing(
        self,
        illuminate = _ALP_DEFAULT.value, picture = _ALP_DEFAULT.value,
        sync_delay = 0, sync_pulse_width = _ALP_DEFAULT.value, trigger_delay = 0
    ):
        self._alp._call(
            "AlpSeqTiming", self._dev._id, self._id,
            c_long(int(illuminate)), c_long(int(picture)),
            c_long(int(sync_delay)), c_long(int(sync_pulse_width)),
            c_long(int(trigger_delay))
        )

    def set_format(self, fmt):
        fmt = self._formats.get(fmt)

        if fmt is None:
            raise ValueError("valid formats are "+", ".join(_formats.keys()))

        self._control(_ALP_DATA_FORMAT, fmt)

    def put(self, offset, n, data):
        data = np.ascontiguousarray(data)

        self._alp._call(
            "AlpSeqPut",
            self._dev._id, self._id, c_long(offset), c_long(n),
            c_void_p(data.__array_interface__["data"][0])
        )

    def set_cycles(self, n):
        self._control(_ALP_SEQ_REPEAT, c_long(n))

    def start(self, continuous = False):
        func_name = "AlpProjStartCont" if continuous else "AlpProjStart"

        self._alp._call(func_name, self._dev._id, self._id)
