# Build the genai-bench benchmark tool image
# Source: https://github.com/sgl-project/genai-bench
FROM python:3.12.12

WORKDIR /genai-bench
ENV PATH="/root/.local/bin:${PATH}"

# Use Aliyun apt mirror
RUN sed -i 's|deb.debian.org|mirrors.aliyun.com|g' /etc/apt/sources.list.d/debian.sources

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    curl \
    gcc \
    git \
    build-essential

# Install pipx and uv using pip
RUN pip install --upgrade pip pipx hatchling wheel -i https://mirrors.aliyun.com/pypi/simple/ && \
    pipx ensurepath && \
    pipx install uv && \
    rm -rf /root/.cache

# Clone genai-bench source and install
RUN git clone --depth 1 \
    https://github.com/sgl-project/genai-bench.git /genai-bench && \
    uv pip install --system -vvv --index-url https://mirrors.aliyun.com/pypi/simple/ /genai-bench

# Clean up unnecessary files to reduce the image size
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/* /var/log/* /root/.cache

ENTRYPOINT ["genai-bench"]
