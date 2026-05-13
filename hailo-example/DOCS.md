# Hailo Example App

Real-time object detection using a USB camera and Hailo-8L.

## Overview

This add-on demonstrates how to use a Hailo-8L AI accelerator for
real-time object detection in Home Assistant. It provides a live
camera view with bounding boxes accessible from the HA sidebar.

## Prerequisites

**The Hailo Service add-on must be installed and running.**
This add-on connects to it via a shared socket to access the
Hailo hardware.

## Supported hardware

- **Platform:** Raspberry Pi 5 (aarch64 only)
- **Accelerator:** Hailo-8L (via Raspberry Pi AI Kit or AI HAT+)
- **Camera:** Any USB webcam or RTSP network camera

## Configuration

### Camera source

Set to your camera device path or RTSP URL:
- USB: `/dev/video0` (default)
- RTSP: `rtsp://user:pass@192.168.1.100:554/stream`

### Confidence threshold

Detections below this confidence score are filtered out.
Range: 0.1 to 1.0 (default: 0.5).

## Web UI

After starting, click "Hailo Detection" in the HA sidebar.
The UI shows:
- Live camera feed with bounding box overlay
- Class labels and confidence scores
- Device status indicator

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| "Waiting for HailoRT service..." forever | Hailo Service add-on not running | Start Hailo Service first |
| Black video / "Camera not available" | Wrong camera_source | Check device path / RTSP URL |
| No detections shown | Threshold too high or model not loaded | Lower threshold, check Hailo Service logs |
| Add-on won't start | Architecture mismatch | This add-on is aarch64 only (RPi 5) |
