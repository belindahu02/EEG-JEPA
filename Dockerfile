# model_training/Dockerfile
FROM ubuntu:22.04
# FROM python:3.9-slim-bullseye # For CPU-only

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Australia/Sydney

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-venv \
    python3.10-distutils \
    python3-pip \
    libjpeg-turbo8-dev \
    libopencv-dev \ 
    pkg-config \
    git \
    tzdata \
    && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime \
    && echo $TZ > /etc/timezone \
    && apt-get clean \
    && [ -e /usr/bin/python3 ] || ln -s /usr/bin/python3.10 /usr/bin/python3 \
    && [ -e /usr/bin/python ] || ln -s /usr/bin/python3.10 /usr/bin/python \
    && [ -e /usr/bin/pip ] || ln -s /usr/bin/pip3 /usr/bin/pip \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
    && python3 --version

RUN ln -sf /usr/bin/python3.10 /usr/bin/python3

# Install PyTorch with custom index

# Then install all other dependencies from requirements.txt via PyPI
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire model_training project folder into /app
# This means /app/src, /app/configs, /app/.project-root, etc.
COPY . .

# Create the .project-root file inside the container
# This is crucial for rootutils to find the project root correctly within the container.
# It should be at /app/.project-root inside the container.
RUN touch .project-root

# Set PYTHONPATH to include /app so Python can find 'src' as a module
ENV PYTHONPATH=/app:$PYTHONPATH

# Define the entry point for your application
# The script to run is now inside 'src/'
ENTRYPOINT ["dora"]
CMD ["run"]
