# Multithread Webcam Access in OpenCV
An elegant implementation of multi-threaded webcam access in OpenCV

# Usage
Execute `python demo.py` to check performance improvement before and after applying the multi-threading class to OpenCV's videoCapture() stream.      

Expect ~10x performance improvement on webcam FPS rate.

# Example
Demo in this repository runs a Caffe trained face detector on webcam stream in real-time. The threading added **173%** improvement on FPS rate after applying the threading.

<img src='screenshot.png'>

# Credit
Heavily inspired by the esteem data scientist Adrian Rosebrock and his <a href='https://www.pyimagesearch.com/2015/12/21/increasing-webcam-fps-with-python-and-opencv/'>article</a>
