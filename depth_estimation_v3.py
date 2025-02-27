#!/usr/bin/env python3
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import argparse
import numpy as np
import cv2
import hailo

from hailo_apps_infra.hailo_rpi_common import (
    get_caps_from_pad,
    get_numpy_from_buffer,
    app_callback_class,
)


# ================================================
# 🔹 User-defined Callback Class
# ================================================
class UserDepthCallback(app_callback_class):
    def __init__(self):
        super().__init__()
        self.frame_skip = 2       # Process every 2nd frame to reduce compute overhead
        self.colormap = cv2.COLORMAP_JET
        self.use_frame = False    # Set True to enable cv2 display


# ================================================
# 🔹 Depth Estimation Callback Function
# ================================================
def app_callback(pad, info, user_data):
    buffer = info.get_buffer()
    if not buffer:
        return Gst.PadProbeReturn.OK

    user_data.increment()
    frame_number = user_data.get_count()

    # Frame Skipping (Improves performance)
    if frame_number % user_data.frame_skip != 0:
        return Gst.PadProbeReturn.OK

    format, width, height = get_caps_from_pad(pad)
    if not (format and width and height):
        return Gst.PadProbeReturn.OK

    # If enabled, retrieve the full RGB frame
    frame = get_numpy_from_buffer(buffer, format, width, height) if user_data.use_frame else None

    # Get the depth metadata from Hailo
    roi = hailo.get_roi_from_buffer(buffer)
    depth_tensors = roi.get_objects_typed(hailo.HAILO_DEPTH)  # Looks for "HAILO_DEPTH" metadata
    if depth_tensors:
        depth_tensor = depth_tensors[0]
        depth_data = np.array(depth_tensor.get_data(), dtype=np.float32)

        # For a 224x224 FastDepth model (adjust if your network differs)
        if depth_data.size == 224 * 224:
            depth_map = depth_data.reshape((224, 224))

            # Normalize and apply colormap for visualization
            depth_normalized = cv2.normalize(
                depth_map, None, 0, 255, cv2.NORM_MINMAX
            ).astype(np.uint8)
            depth_colored = cv2.applyColorMap(depth_normalized, user_data.colormap)

            # Show the depth map using OpenCV
            if user_data.use_frame:
                cv2.imshow("Depth Map", depth_colored)
                cv2.waitKey(1)

    print(f"Processed Frame {frame_number} | Depth Map Checked")
    return Gst.PadProbeReturn.OK


# ================================================
# 🔹 GStreamer Depth Estimation Application
# ================================================
class GStreamerDepthApp:
    def __init__(self, callback, user_data, hef_path, input_source):
        self.callback = callback
        self.user_data = user_data
        self.hef_path = hef_path
        self.input_source = input_source
        self.pipeline = None

    def run(self):
        Gst.init(None)

        # Sample pipeline that takes RGB data at 224x224 -> hailonet -> appsink
        # If you have a camera, change `videotestsrc` to something like `v4l2src device=/dev/video0`
        pipeline_str = (
            f"{self.input_source} ! "
            f"video/x-raw,format=RGB,width=224,height=224,framerate=30/1 ! "
            f"hailonet hef-path={self.hef_path} ! "
            f"videoconvert ! video/x-raw,format=NV12 ! "
            f"appsink name=mysink"
        )

        print("Pipeline string:", pipeline_str)
        self.pipeline = Gst.parse_launch(pipeline_str)

        sink = self.pipeline.get_by_name("mysink")
        if not sink:
            print("Error: Failed to get appsink element")
            return

        # Attach the pad probe callback
        sink_pad = sink.get_static_pad("sink")
        sink_pad.add_probe(Gst.PadProbeType.BUFFER, self.callback, self.user_data)

        # Start GStreamer pipeline
        self.pipeline.set_state(Gst.State.PLAYING)
        bus = self.pipeline.get_bus()

        try:
            while True:
                msg = bus.timed_pop_filtered(
                    Gst.CLOCK_TIME_NONE,
                    Gst.MessageType.ERROR | Gst.MessageType.EOS
                )
                if msg:
                    t = msg.type
                    if t == Gst.MessageType.ERROR:
                        err, dbg = msg.parse_error()
                        print(f"Error from GStreamer: {err}\nDebug: {dbg}")
                        break
                    elif t == Gst.MessageType.EOS:
                        print("End of stream")
                        break
        finally:
            self.pipeline.set_state(Gst.State.NULL)
            # Clean up OpenCV windows
            cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(description="FastDepth Estimation Pipeline for Hailo")
    parser.add_argument("--hef-path", type=str, required=True,
                        help="Path to the HEF file (e.g. fast_depth.hef)")
    parser.add_argument("--use-frame", action="store_true",
                        help="Display the processed depth frame via OpenCV window")
    parser.add_argument("--input", type=str, default="videotestsrc",
                        help="GStreamer input source (default: videotestsrc)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    user_data = UserDepthCallback()
    user_data.use_frame = args.use_frame

    app = GStreamerDepthApp(app_callback, user_data, args.hef_path, args.input)
    app.run()
