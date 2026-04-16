FROM mcr.microsoft.com/devcontainers/python:3.12

# Install Java for MOA
RUN apt-get update --yes \
    && apt-get install -y --no-install-recommends \
        default-jre-headless \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*


RUN python -m pip install --upgrade pip \
    && python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
