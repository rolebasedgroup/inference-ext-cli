# Docker Images

This directory contains Dockerfiles for building project container images.

## Images

| Dockerfile | Image Name | Description |
|---|---|---|
| `autobenchmark-ctl.Dockerfile` | `rolebasedgroup-registry.cn-beijing.cr.aliyuncs.com/dev/rbgs-autobenchmark` | Auto-benchmark controller |
| `benchmark-dashboard.Dockerfile` | `rolebasedgroup-registry.cn-beijing.cr.aliyuncs.com/dev/rbgs-benchmark-dashboard` | Benchmark dashboard |
| `autobenchmark-dashboard.Dockerfile` | `rolebasedgroup-registry.cn-beijing.cr.aliyuncs.com/dev/rbgs-autobenchmark-dashboard` | Auto-benchmark dashboard (React + Nginx) |
| `tools/genai-bench.Dockerfile` | `rolebasedgroup-registry.cn-beijing.cr.aliyuncs.com/dev/rbgs-benchmark-tool-genai` | GenAI-bench benchmark tool |
| `tools/model-downloader-huggingface.Dockerfile` | `rolebasedgroup-registry.cn-beijing.cr.aliyuncs.com/dev/model-downloader-huggingface` | HuggingFace model downloader |
| `tools/model-downloader-modelscope.Dockerfile` | `rolebasedgroup-registry.cn-beijing.cr.aliyuncs.com/dev/model-downloader-modelscope` | ModelScope model downloader |

## Build

All images support multi-architecture builds (`linux/amd64`, `linux/arm64`).

### Core Images

Build from the **repository root**:

```bash
export TAG=v0.1.0

# autobenchmark controller
docker buildx build --platform linux/amd64,linux/arm64 \
  -f docker/autobenchmark-ctl.Dockerfile \
  -t rolebasedgroup-registry.cn-beijing.cr.aliyuncs.com/dev/rbgs-autobenchmark:$TAG \
  --push .

# benchmark dashboard
docker buildx build --platform linux/amd64,linux/arm64 \
  -f docker/benchmark-dashboard.Dockerfile \
  -t rolebasedgroup-registry.cn-beijing.cr.aliyuncs.com/dev/rbgs-benchmark-dashboard:$TAG \
  --push .

# autobenchmark dashboard
docker buildx build --platform linux/amd64,linux/arm64 \
  -f docker/autobenchmark-dashboard.Dockerfile \
  -t rolebasedgroup-registry.cn-beijing.cr.aliyuncs.com/dev/rbgs-autobenchmark-dashboard:$TAG \
  --push .
```

### Tool Images

Build from `docker/tools` directory:

```bash
# genai-bench
docker buildx build --platform linux/amd64,linux/arm64 \
  -f docker/tools/genai-bench.Dockerfile \
  -t rolebasedgroup-registry.cn-beijing.cr.aliyuncs.com/dev/rbgs-benchmark-tool-genai:$TAG \
  --push docker/tools

# model downloader (huggingface)
docker buildx build --platform linux/amd64,linux/arm64 \
  -f docker/tools/model-downloader-huggingface.Dockerfile \
  -t rolebasedgroup-registry.cn-beijing.cr.aliyuncs.com/dev/model-downloader-huggingface:$TAG \
  --push docker/tools

# model downloader (modelscope)
docker buildx build --platform linux/amd64,linux/arm64 \
  -f docker/tools/model-downloader-modelscope.Dockerfile \
  -t rolebasedgroup-registry.cn-beijing.cr.aliyuncs.com/dev/model-downloader-modelscope:$TAG \
  --push docker/tools
```

Replace `<TAG>` with the desired version tag (e.g., `v0.8.0` or `latest`).
