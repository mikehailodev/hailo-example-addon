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

## Switching to URL-based artifact downloads

Currently, `.deb` and `.whl` files are committed in `hailo-example/`.
When public download URLs become available:

1. Delete the committed artifacts:
   - `hailo-example/hailort_4.23.0_arm64.deb`
   - `hailo-example/hailort-4.23.0-cp313-cp313-linux_aarch64.whl`

2. Edit `hailo-example/Dockerfile` — replace the artifact COPY lines with:
   ```dockerfile
   ARG HAILORT_VERSION=4.23.0
   ARG HAILORT_DEB_URL="https://example.com/hailort_${HAILORT_VERSION}_arm64.deb"
   ARG HAILORT_WHL_URL="https://example.com/hailort-${HAILORT_VERSION}-cp313-cp313-linux_aarch64.whl"

   # Stage 1: download and extract .deb
   FROM debian:bookworm-slim AS extractor
   RUN apt-get update && apt-get install -y --no-install-recommends curl
   ARG HAILORT_DEB_URL
   RUN curl -fsSL -o /tmp/hailort.deb "${HAILORT_DEB_URL}" \
       && mkdir /tmp/hailort && dpkg-deb -x /tmp/hailort.deb /tmp/hailort

   # Stage 2: Python runtime
   FROM python:3.13-slim-bookworm
   COPY --from=extractor /tmp/hailort/usr/lib/libhailort.so* /usr/lib/
   RUN ldconfig
   ARG HAILORT_WHL_URL
   RUN pip install --no-cache-dir "${HAILORT_WHL_URL}" numpy opencv-python-headless flask
   ```

3. Remove artifacts from git history:
   ```bash
   git filter-repo --path hailo-example/hailort_4.23.0_arm64.deb --invert-paths
   git filter-repo --path hailo-example/hailort-4.23.0-cp313-cp313-linux_aarch64.whl --invert-paths
   git push --force
   ```

## License

MIT
