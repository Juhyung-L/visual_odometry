FROM osrf/ros:humble-desktop-full-jammy

# USE BASH
SHELL ["/bin/bash", "-c"]

# RUN LINE BELOW TO REMOVE debconf ERRORS (MUST RUN BEFORE ANY apt-get CALLS)
RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections

WORKDIR /home/dev_ws

# install opencv, and eigen3
RUN apt-get update && apt-get install -q -y --no-install-recommends \
   libopencv-dev \
   libeigen3-dev \
   && rm rm -rf /var/lib/apt/lists/*

# personal packages
RUN apt-get update && apt-get install -q -y --no-install-recommends \
   vim \
   gdb \
   gdbserver \
   python3 \
   pip \
   && rm -rf /var/lib/apt/lists/*

# setup bashrc settings
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc && \
    echo "alias b='source /home/dev_ws/install/setup.bash'" >> ~/.bashrc
