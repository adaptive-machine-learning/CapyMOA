FROM mcr.microsoft.com/devcontainers/python:3.12

# Install Java for MOA
RUN apt-get update --yes \
    && apt-get install -y --no-install-recommends \
        default-jre-headless \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*


COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/
