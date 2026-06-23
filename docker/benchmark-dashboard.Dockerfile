# Build the benchmark-dashboard binary
FROM golang:1.26 AS builder
ARG TARGETOS
ARG TARGETARCH

ARG GOPROXY
ARG GOPRIVATE
ARG GOSUMDB

ENV GOPROXY=${GOPROXY} \
    GOPRIVATE=${GOPRIVATE} \
    GOSUMDB=${GOSUMDB}

WORKDIR /workspace
ADD . /workspace

# Build
RUN CGO_ENABLED=0 GOOS=${TARGETOS:-linux} GOARCH=${TARGETARCH:-amd64} \
    go build -mod vendor -o benchmark-dashboard ./ui/benchmark/

# Use distroless as minimal base image
FROM alpine:3.22

WORKDIR /
COPY --from=builder /workspace/benchmark-dashboard /benchmark-dashboard
USER 65532:65532

ENTRYPOINT ["/benchmark-dashboard"]
