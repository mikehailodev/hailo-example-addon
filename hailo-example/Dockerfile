ARG BUILD_FROM=python:3.13-slim-bookworm
FROM ${BUILD_FROM}

ENV DEBIAN_FRONTEND=noninteractive

# Install HailoRT client library from .deb (libhailort + hailortcli)
COPY hailort_4.23.0_arm64.deb /tmp/hailort.deb
RUN apt-get update && apt-get install -y --no-install-recommends bash \
    && dpkg -i /tmp/hailort.deb || true \
    && apt-get install -yf --no-install-recommends \
    && rm -f /tmp/hailort.deb \
    && rm -rf /var/lib/apt/lists/* \
    && ldconfig

# Install Python bindings + numpy
COPY hailort-4.23.0-cp313-cp313-linux_aarch64.whl /tmp/
RUN pip install --no-cache-dir /tmp/hailort-4.23.0-cp313-cp313-linux_aarch64.whl numpy \
    && rm -f /tmp/*.whl

COPY detect.py /detect.py
COPY run.sh /run.sh
RUN chmod a+x /run.sh

LABEL \
    io.hass.version="1.0.0" \
    io.hass.type="addon" \
    io.hass.arch="aarch64|amd64"

CMD ["/run.sh"]
