# Dockerfile: Build an ARM64-compatible image with OpenCV and your VO dependencies

FROM arm64v8/ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Install OpenCV (optional: use precompiled .deb for speed)
RUN apt-get update && apt-get install -y libopencv-dev

# Install basic packages
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    unzip \
    pkg-config \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libgtk2.0-dev \
    libgtk-3-dev \
    libcanberra-gtk* \
    libatlas-base-dev \
    gfortran \
    python3-dev \
    python3-pip \
    libeigen3-dev

# Create app directory
WORKDIR /app

# Copy your project code into the container
COPY ../ /app/

# Build your code
# RUN mkdir -p build && cd build && cmake .. && make -j4

CMD ["bash"]

