# Use a slim Python base image to keep the image size small
FROM python:3.9-slim-bullseye

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file first to leverage Docker layer caching
COPY requirements.txt .

# Install Python dependencies
# Use --no-cache-dir to avoid storing pip's cache in the image, reducing size
# Consider installing build-essential if MNE or other libraries need compilation,
# though pip often provides pre-built wheels.
# apt-get update && apt-get install -y build-essential
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    libblas3 \
    liblapack3 \
    # Add other system-level dependencies for MNE if needed
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir -r requirements.txt

# If you need GPU support for PyTorch:
# You'll need a different base image (e.g., nvidia/cuda) and a specific PyTorch installation.
# This example assumes CPU-only or that your remote server has CUDA setup and you install appropriate torch version.
# For CPU only, `pip install torch` is generally fine.

# Copy your application code into the container
COPY . .

# Define a default command to run when the container starts
# This uses 'fire' which is good for command-line arguments.
# This makes 'python' the permanent executable for the container
ENTRYPOINT ["python"]

# This provides the default script and arguments to the ENTRYPOINT (python)
CMD ["eeg_to_lms.py"]
