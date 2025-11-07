# Use NVIDIA CUDA base image with cuDNN
FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

# Set working directory
WORKDIR /app

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Australia/Sydney

# Install Python, pip, and common system dependencies
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
    && ln -sf /usr/bin/python3.10 /usr/bin/python3 \
    && ln -sf /usr/bin/pip3 /usr/bin/pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your code
COPY . .

# Set PYTHONPATH so your modules can be found
ENV PYTHONPATH=/app:$PYTHONPATH

# Run your script
CMD ["python3", "supervised/plot_results.py"]
