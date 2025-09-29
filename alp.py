from .error import AlpError
from ._consts import *
from ._structs import *
from ctypes import CDLL, c_long, c_void_p, byref
from pathlib import Path
import numpy as np



class Alp:
    _lib = None

    def __init__(self, library_path = None):
        if self._lib is not None:
            raise RuntimeError("ALP already initialised")

        if library_path is None:
            paths = Path("/").glob("Program Files*/ALP*/**/alp*.dll")
            fail_msg = "could not load any ALP libraries discovered" 
        else:
            paths = [library_path]
            fail_msg = "could not load ALP library given"

        for path in paths:
            try:
                type(self)._lib = CDLL(path)
                break
            except OSError:
                pass
        else:
            raise RuntimeError(fail_msg)

    def __enter__(self):
        return self

    def __exit__(self, e_type, e_value, e_trace):
        self.close()

    def close(self):
        type(self)._lib = None

    def _call(self, func_name, *args):
        ret = self._lib[func_name](*args)

        if ret != ALP_OK.value: raise AlpError(ret)

    def open_device(self, device_num = ALP_DEFAULT.value):
        dev = AlpDevice()
        dev._id = c_long(0)
        dev._alp = self

        self._call(
            "AlpDevAlloc",
            c_long(device_num), c_long(0), byref(dev._id)
        )

        return dev

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

    def __enter__(self):
        return self

    def __exit__(self, e_type, e_value, e_trace):
        self.close()

    def close(self):
        self.halt()
        self._alp._call("AlpDevFree", self._id)

    def get_display_size(self):
        width = self._inquire(ALP_DEV_DISPLAY_WIDTH).value
        height = self._inquire(ALP_DEV_DISPLAY_HEIGHT).value

        return width, height

    def get_proj_progress(self):
        info = AlpProjProgress()

        self._alp._call(
            "AlpProjInquireEx",
            self._id, ALP_PROJ_PROGRESS, byref(info)
        )

        return info

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
        code = self._inquire(ALP_PROJ_STATE, proj = True)

        if code.value == ALP_PROJ_ACTIVE.value: return True
        elif code.value == ALP_PROJ_IDLE.value: return False
        else:
            raise RuntimeError("AlpProjInquire gave invalid ALP_PROJ_STATE")

    def set_trigger(self, trigger_type):
        self._control(ALP_PROJ_STEP, trigger_type.value)

class AlpSequence:
    def _control(self, param, value):
        self._alp._call(
            "AlpSeqControl",
            self._dev._id, self._id, param, value
        )

    def __enter__(self):
        return self

    def __exit__(self, e_type, e_value, e_trace):
        self.free()

    def free(self):
        self._dev.halt()
        self._alp._call("AlpSeqFree", self._dev._id, self._id)

    def set_timing(
        self,
        illuminate = ALP_DEFAULT.value, picture = ALP_DEFAULT.value,
        sync_delay = 0, sync_pulse_width = ALP_DEFAULT.value, trigger_delay = 0
    ):
        self._alp._call(
            "AlpSeqTiming", self._dev._id, self._id,
            c_long(int(illuminate)), c_long(int(picture)),
            c_long(int(sync_delay)), c_long(int(sync_pulse_width)),
            c_long(int(trigger_delay))
        )

    def set_format(self, fmt):
        self._control(ALP_DATA_FORMAT, fmt.value)

    def put(self, offset, n, data):
        data = np.ascontiguousarray(data, dtype = "u1")

        self._alp._call(
            "AlpSeqPut",
            self._dev._id, self._id, c_long(offset), c_long(n),
            c_void_p(data.__array_interface__["data"][0])
        )

    def set_cycles(self, n):
        self._control(ALP_SEQ_REPEAT, c_long(n))

    def start(self, continuous = False):
        func_name = "AlpProjStartCont" if continuous else "AlpProjStart"

        self._alp._call(func_name, self._dev._id, self._id)
