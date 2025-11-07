# GPU-enabled base image with CUDA 11.8 and cuDNN 8 (Ubuntu 22.04)
#FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Set up environment
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Australia/Sydney
WORKDIR /app

# Set CUDA environment variables
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
ENV TF_ENABLE_ONEDNN_OPTS=0
ENV TF_XLA_FLAGS="--tf_xla_enable_xla_devices=false"  
# disable XLA JIT to avoid compilation issues

# Install Python 3.10 and basic dependencies
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
 && rm -rf /var/lib/apt/lists/*

# Ensure python and pip commands point to python3.10
RUN ln -sf /usr/bin/python3.10 /usr/bin/python3 && \
    ln -sf /usr/bin/pip3 /usr/bin/pip && \
    python3 --version && pip --version

# Copy requirements first (for layer caching)
COPY requirements.txt .

# Upgrade pip and install dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of your project
COPY . .

# Create .project-root (for rootutils)
RUN touch .project-root

# Ensure Python can find your modules
ENV PYTHONPATH=/app:$PYTHONPATH

# Entry point
ENTRYPOINT ["dora"]
CMD ["run"]

