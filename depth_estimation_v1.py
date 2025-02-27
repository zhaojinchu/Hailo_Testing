#!/usr/bin/env python3
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import argparse
import os
import numpy as np
import cv2
import hailo

from hailo_apps_infra.hailo_rpi_common import (
    get_caps_from_pad,
    get_numpy_from_buffer,
    app_callback_class,
)

# =============================================================================
# User-defined callback class
# =============================================================================
class user_app_callback_class(app_callback_class):
    def __init__(self):
        super().__init__()
        # You can add custom variables here if needed

# =============================================================================
# Depth Estimation Callback Function
# =============================================================================
def app_callback(pad, info, user_data):
    buffer = info.get_buffer()
    if buffer is None:
        return Gst.PadProbeReturn.OK

    user_data.increment()
    frame_number = user_data.get_count()

    # Retrieve video frame from buffer (if enabled)
    format, width, height = get_caps_from_pad(pad)
    frame = None
    if user_data.use_frame and format is not None and width is not None and height is not None:
        frame = get_numpy_from_buffer(buffer, format, width, height)

    # Get the ROI from the buffer that holds the model outputs
    roi = hailo.get_roi_from_buffer(buffer)
    # Retrieve the output tensor.
    # (We assume the fast_depth model output is attached as a tensor.
    # Replace HAILO_TENSOR with the proper constant if needed.)
    depth_outputs = roi.get_objects_typed(hailo.HAILO_TENSOR)

    if depth_outputs and len(depth_outputs) > 0:
        tensor_obj = depth_outputs[0]
        # Hypothetical API: get_data() returns the raw depth values as a flat list.
        # We expect an output shape of 224x224x1.
        data = np.array(tensor_obj.get_data(), dtype=np.float32)
        if data.size != 224 * 224:
            print(f"Unexpected tensor size: {data.size}")
        else:
            depth_map = data.reshape((224, 224))
            # Normalize the depth map for visualization
            depth_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
            depth_normalized = depth_normalized.astype(np.uint8)
            # Apply a colormap to produce a pseudo-colored depth image
            depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
            if user_data.use_frame:
                user_data.set_frame(depth_colored)
    else:
        print("No depth tensor found in ROI.")

    print(f"Processed frame {frame_number} for depth estimation.")
    return Gst.PadProbeReturn.OK

# =============================================================================
# GStreamer Pipeline Application for Depth Estimation
# =============================================================================
class GStreamerDepthEstimationApp:
    def __init__(self, callback, user_data, hef_path, input_source):
        self.callback = callback
        self.user_data = user_data
        self.hef_path = hef_path
        self.input_source = input_source  # e.g. "videotestsrc" or "filesrc location=yourvideo.mp4"
        self.pipeline = None

    def run(self):
        Gst.init(None)
        # Construct the pipeline string.
        # The pipeline reads from an input source, forces the video format to RGB at 224x224,
        # passes frames to the hailonet element (loading your HEF file),
        # and then sends output to an appsink.
        pipeline_str = (
            f"{self.input_source} ! video/x-raw,format=RGB,width=224,height=224,framerate=30/1 ! "
            f"hailonet hef-path={self.hef_path} ! videoconvert ! appsink name=mysink"
        )
        print("Pipeline string:", pipeline_str)
        self.pipeline = Gst.parse_launch(pipeline_str)

        # Get the appsink element and add a buffer probe to attach our callback.
        sink = self.pipeline.get_by_name("mysink")
        if not sink:
            print("Failed to get appsink element from pipeline!")
            return

        # Add a probe on the sink pad so our callback gets invoked for each buffer.
        sink_pad = sink.get_static_pad("sink")
        sink_pad.add_probe(Gst.PadProbeType.BUFFER, self.callback, self.user_data)

        # Set the pipeline to PLAYING and run the main loop.
        self.pipeline.set_state(Gst.State.PLAYING)
        bus = self.pipeline.get_bus()
        try:
            while True:
                msg = bus.timed_pop_filtered(
                    Gst.CLOCK_TIME_NONE, Gst.MessageType.ERROR | Gst.MessageType.EOS
                )
                if msg:
                    t = msg.type
                    if t == Gst.MessageType.ERROR:
                        err, debug = msg.parse_error()
                        print("Error received from element %s: %s" % (msg.src.get_name(), err))
                        print("Debugging information: %s" % debug)
                        break
                    elif t == Gst.MessageType.EOS:
                        print("End-Of-Stream reached.")
                        break
        finally:
            self.pipeline.set_state(Gst.State.NULL)

# =============================================================================
# Argument Parsing and Main Routine
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Fast Depth Estimation Pipeline")
    parser.add_argument("--hef-path", type=str, required=True,
                        help="Path to the HEF file for the fast_depth model")
    parser.add_argument("--use-frame", action="store_true",
                        help="Display the output frame (requires an appropriate display handler)")
    parser.add_argument("--input", type=str, default="videotestsrc",
                        help="Input source for video (default: videotestsrc)")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    # Create an instance of the user callback class.
    user_data = user_app_callback_class()
    user_data.use_frame = args.use_frame

    # Instantiate and run the GStreamer depth estimation pipeline.
    app = GStreamerDepthEstimationApp(app_callback, user_data, args.hef_path, args.input)
    app.run()
