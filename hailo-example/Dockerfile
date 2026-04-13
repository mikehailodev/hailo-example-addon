FROM python:3.11-slim-bookworm

ARG TARGETARCH=arm64
ARG HAILORT_VERSION=4.23.0

# Install HailoRT client libraries (pre-built by frigate-nvr).
# These are the SAME libraries Frigate uses — no device access needed,
# only the gRPC client that talks to hailort_service.
RUN apt-get update && apt-get install -y --no-install-recommends wget && \
    if [ "${TARGETARCH}" = "amd64" ]; then arch="x86_64"; else arch="aarch64"; fi && \
    wget -qO- \
      "https://github.com/frigate-nvr/hailort/releases/download/v${HAILORT_VERSION}/hailort-debian12-${TARGETARCH}.tar.gz" \
      | tar -C / -xzf - && \
    pip install --no-cache-dir \
      "https://github.com/frigate-nvr/hailort/releases/download/v${HAILORT_VERSION}/hailort-${HAILORT_VERSION}-cp311-cp311-linux_${arch}.whl" && \
    pip install --no-cache-dir numpy && \
    apt-get remove -y wget && apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

COPY detect.py /detect.py
COPY run.sh /run.sh
RUN chmod a+x /run.sh

LABEL \
    io.hass.version="1.0.0" \
    io.hass.type="addon" \
    io.hass.arch="aarch64|amd64"

CMD ["/run.sh"]
