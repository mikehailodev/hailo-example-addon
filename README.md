# Hailo Example Add-on for Home Assistant

Real-time object detection demo using a USB camera and Hailo-8L
via the HailoRT multi-process service. Displays live video with
bounding boxes through the Home Assistant ingress UI.

## What it does

- Captures frames from a USB camera (or RTSP stream)
- Runs YOLOv6n inference on Hailo-8L via the multi-process service
- Serves a web UI with live bounding box overlay (accessible via HA sidebar)

## Requirements

- **Hailo Service Add-on** — must be installed and running first
  (provides the HailoRT multi-process daemon)
- USB camera (e.g. `/dev/video0`) or RTSP stream URL
- Raspberry Pi 5 with Hailo-8L (AI Kit / AI HAT+)

## Architecture: aarch64 only

This add-on targets **Raspberry Pi 5** (aarch64) exclusively.
It is a community demo/reference, not a production multi-arch add-on.
For x86-64 + Hailo, use Frigate with `type: hailo` detector instead.

## Installation

1. Install and start the **Hailo Service** add-on first
2. Add this repository:
   **Settings → Add-ons → Add-on Store → ⋮ → Repositories**
   ```
   https://github.com/mikehailodev/hailo-example-addon
   ```
3. Install "Hailo Example App"
4. Configure camera source in the add-on options
5. Start the add-on — it appears in the HA sidebar as "Hailo Detection"

## Configuration

| Option | Default | Description |
|--------|---------|-------------|
| `camera_source` | `/dev/video0` | USB camera device or RTSP URL |
| `confidence_threshold` | `0.5` | Minimum detection confidence (0.1–1.0) |

### RTSP example:
```yaml
camera_source: "rtsp://192.168.1.100:8554/cam1"
confidence_threshold: 0.4
```

## How it works

```
┌────────────────────────────────────────┐
│        Hailo Example Add-on            │
│                                        │
│  USB cam → OpenCV → Hailo inference    │
│         → Flask web UI (port 8099)     │
│         → HA ingress (sidebar panel)   │
└────────────────┬───────────────────────┘
                 │ gRPC (unix socket)
                 ▼
    /share/hailo/hailort_service.sock
                 │
         ┌───────┴────────┐
         │ Hailo Service   │
         │ /dev/hailo0     │
         └─────────────────┘
```

## License

MIT
