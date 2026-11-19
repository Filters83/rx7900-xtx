# From the repository root:
# docker build -f nariq/Dockerfile --target artifacts --output type=local,dest=nariq-dist .
# docker build -f nariq/Dockerfile -t nari-llama.cpp:local .
# The release image, from the release's two Linux archives and SHA256SUMS (the build context is their folder):
# docker build -f nariq/Dockerfile --target runtime-release -t nari-llama.cpp:<tag> <archive folder>
ARG CUDA_VERSION=12.8.2
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04 AS build
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates ninja-build build-essential python3-pip \
    && pip3 install --no-cache-dir cmake==3.31.6 && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY . .
ARG NARIQ_SOURCE_REVISION=unknown
ENV NARIQ_SOURCE_REVISION=${NARIQ_SOURCE_REVISION}
ARG JOBS=4
ARG CUDA_ARCHS
RUN JOBS="$JOBS" CUDA_ARCHS="$CUDA_ARCHS" bash nariq/build_linux.sh /dist
RUN mkdir /app && for a in /dist/*.tar.gz; do tar -xzf "$a" -C /app --strip-components=1; done

FROM scratch AS artifacts
COPY --from=build /dist/ /

# The archives exactly as released: checked against SHA256SUMS, then unpacked into /app
FROM nvidia/cuda:${CUDA_VERSION}-base-ubuntu22.04 AS release-files
COPY SHA256SUMS *.tar.gz /release/
RUN cd /release && test "$(grep -c '\.tar\.gz$' SHA256SUMS)" = 2 \
    && ls nari-llama.cpp-*-bin-linux-cuda-*-x64.tar.gz cudart-nari-llama.cpp-*-linux-cuda-*-x64.tar.gz \
    && sha256sum --strict -c SHA256SUMS \
    && mkdir /app && for a in *.tar.gz; do tar -xzf "$a" -C /app --strip-components=1; done

FROM nvidia/cuda:${CUDA_VERSION}-base-ubuntu22.04 AS runtime-base
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl && rm -rf /var/lib/apt/lists/*
ENV LD_LIBRARY_PATH=/app:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 LLAMA_ARG_HOST=0.0.0.0 LLAMA_ARG_PORT=8080
EXPOSE 8080
HEALTHCHECK --start-period=10m CMD curl -fs http://localhost:8080/health || exit 1
ENTRYPOINT ["/app/llama-server"]

FROM runtime-base AS runtime-release
COPY --from=release-files /app /app

FROM runtime-base AS runtime
COPY --from=build /app /app
