FROM registry.roqs.basf.net/rormula/rust-multi-python:latest

USER root
COPY pyrormula /home/rust/pyrormula
COPY rormula /home/rust/rormula
WORKDIR /home/rust/pyrormula
RUN python3 -m venv .venv
RUN . .venv/bin/activate
RUN pip3 install maturin==0.14.16 pytest formulaic 
RUN maturin develop --target x86_64-unknown-linux-gnu
RUN chown -R rust /home/rust
USER rust
RUN curl -O https://ziglang.org/download/0.10.1/zig-linux-x86_64-0.10.1.tar.xz
RUN tar -xf zig-linux-x86_64-0.10.1.tar.xz
