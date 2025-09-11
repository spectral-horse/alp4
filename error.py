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
        if code == 0:
            raise ValueError("code 0 is ALP_OK, not an error")

        super().__init__(self._err_codes.get(code, "UNKNOWN"))
