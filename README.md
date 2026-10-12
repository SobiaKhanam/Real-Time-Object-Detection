# Real-Time Object Detection Using YOLO on NVIDIA Jetson Nano

### Pretrained Deep Learning Model Deployment for Edge Computer Vision

Developed as part of an internship research project at TTL in 2021, this project implemented real-time object detection using a pretrained YOLO (You Only Look Once) model on an NVIDIA Jetson Nano. Using a live camera feed, the system detects multiple objects in the surrounding environment and displays their class labels, bounding boxes, and confidence scores.

## Project Demo

[Watch the full demonstration](https://drive.google.com/file/d/1cW3iCR8bDX0XdN4cC0nJ7GgJOblRxLpL/view?usp=sharing)

## Project Overview

The project explored deploying a pretrained deep learning model on an edge computing device for real-time visual recognition. Camera frames were processed to identify objects such as bottles, cups, and bowls, with detection results overlaid on the live video feed.

Rather than training a model from scratch, the project focused on using an existing pretrained model for inference on embedded hardware.

## Key Features

- Real-time object detection from a live camera feed
- Pretrained YOLO model for object recognition
- Bounding-box visualization around detected objects
- Object class labels and confidence scores
- Edge inference using NVIDIA Jetson Nano

## Technology Stack

- Python
- YOLO (pretrained object detection model)
- NVIDIA Jetson Nano
- Camera-based video input
- Computer vision and deep learning inference

## Project Background

- **Project type:** Internship Research Project
- **Organization:** Techrotics Technologies Lab (TTL)
- **Year:** 2021
- **Focus areas:** Object Detection, Deep Learning, Computer Vision, and Edge AI

## References

- [NVIDIA jetson-inference](https://github.com/dusty-nv/jetson-inference)
- [NVIDIA object detection example](https://github.com/dusty-nv/jetson-inference/blob/master/python/examples/my-detection.py)
