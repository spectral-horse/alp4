from ctypes import Structure, c_ulong



class AlpProjProgress(Structure):
    _fields_ = [
        ("queue_id", c_ulong),
        ("sequence_id", c_ulong),
        ("waiting_sequences", c_ulong),
        ("cycles_remaining", c_ulong),
        ("cycles_remaining_underflow", c_ulong),
        ("frames_remaining", c_ulong),
        ("picture_time", c_ulong),
        ("frames_total", c_ulong),
        ("flags", c_ulong)
    ]

    @property
    def frame_index(self):
        return (self.frames_total-1)-self.frames_remaining
