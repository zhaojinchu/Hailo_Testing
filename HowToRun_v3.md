Install Hailo GStreamer plugin and ensure hailonet is recognized:

For example, you can verify using:

gst-inspect-1.0 hailo

If the plugin is missing, reinstall or re-check your Hailo software environment.
Pass the correct HEF (e.g. fast_depth.hef) and your desired GStreamer source:

If you’re using a USB camera on /dev/video0, do:

python depth_estimation.py --hef-path fast_depth.hef --use-frame \
  --input "v4l2src device=/dev/video0"

If you want to see a synthetic test pattern, omit --use-frame or keep videotestsrc.
Confirm you have a GUI session so OpenCV can display the window:

For example, if you’re running from the Pi’s desktop or via SSH with X forwarding.
Press Ctrl+C in the terminal or close the window to end the pipeline.


Depending on your camera input format, you may need an additional videoconvert or videoscale upstream. For example:
--input "v4l2src device=/dev/video0 ! videoconvert ! videoscale ! video/x-raw,format=RGB,width=224,height=