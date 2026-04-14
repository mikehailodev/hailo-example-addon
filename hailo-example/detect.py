#!/usr/bin/env python3
"""
Hailo Example Client — Blueprint for multi-container Hailo inference.

Architecture:
  ┌──────────────────────┐        ┌──────────────────────────┐
  │  This container       │        │  Hailo Service container  │
  │  (hailo_example)      │        │  (hailo_service)          │
  │                       │        │                           │
  │  detect.py            │        │  hailort_service daemon   │
  │    │                  │        │    │                      │
  │    ├─ libhailort.so ──┼─gRPC──▶│    └─── /dev/hailo0      │
  │    │  (client mode)   │  Unix  │         (PCIe device)     │
  │    │                  │ socket │                           │
  │    └─ SHM data ───────┼───────▶│    SHM data               │
  │                       │ shared │                           │
  └──────────────────────┘  IPC   └──────────────────────────┘

Key settings that make this work:
  - Both add-ons: host_ipc: true     (shares /dev/shm for inference data)
  - Both add-ons: map: share         (shares /share/ for the Unix socket)
  - Service add-on: devices: /dev/hailo0
  - Client add-on: NO device access needed

Client-side code pattern:
  1. Set HAILO_SOCK_PATH to the shared socket directory
  2. Create VDevice with multi_process_service=True
  3. Use the same group_id across all clients
  4. Load HEF, run inference — the service handles device scheduling
"""

import logging
import os
import sys
import urllib.request

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Default YOLOv6n models from Hailo Model Zoo
MODEL_URLS = {
    "hailo8": "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.14.0/hailo8/yolov6n.hef",
    "hailo8l": "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.14.0/hailo8l/yolov6n.hef",
}
MODEL_DIR = "/data/models"
MODEL_NAME = "yolov6n.hef"


def download_model(url: str, dest: str):
    if os.path.exists(dest):
        logger.info("Model cached: %s", dest)
        return
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    logger.info("Downloading model from %s ...", url)
    urllib.request.urlretrieve(url, dest)
    logger.info("Saved to %s", dest)


def main():
    try:
        from hailo_platform import (
            VDevice,
            HEF,
            ConfigureParams,
            InputVStreamParams,
            OutputVStreamParams,
            InferVStreams,
            HailoStreamInterface,
            HailoSchedulingAlgorithm,
        )
    except ImportError:
        logger.error("hailo_platform not found. Is pyHailoRT installed?")
        sys.exit(1)

    sock_path = os.environ.get("HAILO_SOCK_PATH", "(not set)")
    logger.info("=" * 60)
    logger.info("Hailo Example Client — Multi-Container Blueprint")
    logger.info("=" * 60)
    logger.info("HAILO_SOCK_PATH = %s", sock_path)

    # ── Step 1: Connect to HailoRT service ──────────────────────
    logger.info("Connecting to HailoRT service...")

    params = VDevice.create_params()
    params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
    params.multi_process_service = True
    params.group_id = "SHARED"

    vdevice = VDevice(params)
    logger.info("Connected to HailoRT service!")

    # ── Step 2: Download model ──────────────────────────────────
    # Default to hailo8; override by setting HAILO_ARCH env var
    arch = os.environ.get("HAILO_ARCH", "hailo8l")
    model_url = MODEL_URLS.get(arch, MODEL_URLS["hailo8"])
    model_path = os.path.join(MODEL_DIR, MODEL_NAME)
    download_model(model_url, model_path)

    # ── Step 3: Load HEF and configure ──────────────────────────
    logger.info("Loading model: %s", model_path)
    hef = HEF(model_path)

    configure_params = ConfigureParams.create_from_hef(
        hef=hef, interface=HailoStreamInterface.PCIe
    )
    network_group = vdevice.configure(hef, configure_params)[0]

    input_vstreams_params = InputVStreamParams.make(network_group)
    output_vstreams_params = OutputVStreamParams.make(network_group)

    input_info = hef.get_input_vstream_infos()[0]
    logger.info("Input: name=%s shape=%s", input_info.name, input_info.shape)

    output_infos = hef.get_output_vstream_infos()
    for o in output_infos:
        logger.info("Output: name=%s shape=%s", o.name, o.shape)

    # ── Step 4: Create test input ───────────────────────────────
    h, w, c = input_info.shape
    logger.info("Creating test image (%dx%dx%d)...", w, h, c)
    test_image = np.random.randint(0, 255, (h, w, c), dtype=np.uint8)
    # Add batch dimension
    batch = np.expand_dims(test_image, axis=0)

    # ── Step 5: Run inference ───────────────────────────────────
    logger.info("Running inference...")
    with InferVStreams(
        network_group, input_vstreams_params, output_vstreams_params
    ) as pipeline:
        input_data = {input_info.name: batch}
        results = pipeline.infer(input_data)

    # ── Step 6: Print results ───────────────────────────────────
    logger.info("Inference complete!")
    for name, data in results.items():
        arr = np.array(data)
        logger.info("  %s: shape=%s dtype=%s min=%.4f max=%.4f",
                     name, arr.shape, arr.dtype, arr.min(), arr.max())

    logger.info("=" * 60)
    logger.info("SUCCESS — multi-container Hailo inference works!")
    logger.info("=" * 60)

    # Clean up
    vdevice.release()


if __name__ == "__main__":
    main()
