FROM rust:1-slim

WORKDIR /workspace
COPY Cargo.toml ./
COPY src ./src

RUN cargo test --all-targets

CMD ["cargo", "test", "--all-targets"]
