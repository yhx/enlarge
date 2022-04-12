FROM nvidia/cuda:11.2.2-devel-ubuntu20.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update
RUN apt-get install apt-utils
RUN apt-get install -y gcc cmake libopenmpi-dev git
RUN git clone --recursive https://github.com/yhx/enlarge.git
RUN git clone https://github.com/nest/nest-simulator.git
RUN apt-get install -y python3-dev libblas-dev openssh-server python3-pip  cython libgsl-dev libltdl-dev libncurses-dev libreadline-dev  openmpi-bin libopenmpi-dev
