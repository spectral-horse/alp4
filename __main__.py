from . import *




alp = Alp("./alp4395.dll")

try:
    dmd = alp.open_device()
except AlpError:
    print("Couldn't open DMD:", *e.args)
    exit(1)

width, height = dmd.get_display_size()

print(f"Found DMD of size {width}x{height}")
