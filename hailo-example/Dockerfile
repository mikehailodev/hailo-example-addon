# Stage 1: Extract .deb on a debian image (has dpkg-deb)
FROM debian:bookworm-slim AS extractor
COPY hailort_4.23.0_arm64.deb /tmp/hailort.deb
RUN mkdir /tmp/hailort && dpkg-deb -x /tmp/hailort.deb /tmp/hailort

# Stage 2: Python runtime
FROM python:3.13-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive

# Copy only the shared library from extractor
COPY --from=extractor /tmp/hailort/usr/lib/libhailort.so* /usr/lib/
RUN ldconfig

# Install Python bindings + deps (netifaces needs gcc to build)
COPY hailort-4.23.0-cp313-cp313-linux_aarch64.whl /tmp/
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc libc6-dev libgl1 libglib2.0-0 \
    && pip install --no-cache-dir \
        /tmp/hailort-4.23.0-cp313-cp313-linux_aarch64.whl \
        numpy opencv-python-headless flask \
    && rm -f /tmp/*.whl \
    && apt-get remove -y gcc libc6-dev && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

COPY server.py /app/server.py
COPY static/ /app/static/
COPY run.sh /run.sh
RUN chmod a+x /run.sh

LABEL \
    io.hass.version="1.1.0" \
    io.hass.type="addon" \
    io.hass.arch="aarch64"

CMD ["/run.sh"]
