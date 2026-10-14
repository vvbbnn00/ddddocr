# syntax=docker/dockerfile:1
FROM rust:1.96-bookworm AS build
WORKDIR /app
ENV CARGO_REGISTRIES_CRATES_IO_PROTOCOL=sparse
RUN apt-get update && apt-get install -y --no-install-recommends cmake pkg-config libssl-dev \
    && rm -rf /var/lib/apt/lists/*
COPY Cargo.toml Cargo.lock ./
COPY src ./src
COPY model ./model
COPY initialize.json list.json ./
# Refuse Git dependencies before Cargo can fetch any repository.
RUN ! grep -q '^source = "git+' Cargo.lock \
    && cargo build --release --locked

FROM debian:bookworm-slim AS runtime
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates libssl3 libgomp1 libstdc++6 curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --system --uid 10001 --no-create-home ddddocr
WORKDIR /app
COPY --from=build /app/target/release/ddddocr /usr/local/bin/ddddocr
USER 10001:10001
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=30s --retries=3 \
    CMD curl --fail --silent http://127.0.0.1:8000/health || exit 1
ENTRYPOINT ["ddddocr"]
CMD ["--address", "0.0.0.0:8000"]
