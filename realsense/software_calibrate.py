import pyrealsense2 as rs

ctx = rs.context()
device_list = ctx.query_devices()
if len(device_list) == 0:
    raise Exception("No RealSense device connected")

dev = device_list[0]  # Get the first connected device

# This is the correct call (on the device object, not device list)
calib_dev = dev.as_auto_calibrated_device()