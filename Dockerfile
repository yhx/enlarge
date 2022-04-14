FROM nvidia/cuda:11.2.2-devel-ubuntu20.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update
RUN apt-get install apt-utils
RUN apt-get install -y gcc cmake libopenmpi-dev git
RUN git clone --recursive https://github.com/yhx/enlarge.git
RUN git clone https://github.com/nest/nest-simulator.git
RUN git clone https://github.com/spack/spack.git /opt/spack
RUN apt-get install -y python3-dev libblas-dev openssh-server python3-pip  cython libgsl-dev libltdl-dev libncurses-dev libreadline-dev  openmpi-bin libopenmpi-dev python3-pytest python3-pytest-xdist python3-pytest-timeout dnsutils python3-numpy python3-scipy python3-matplotlib
RUN sed -i '/UsePAM yes/cUsePAM no' /etc/ssh/sshd_config && sed -i '/#PermitRootLogin prohibit-password/cPermitRootLogin yes' /etc/ssh/sshd_config && sed -i '/#PubkeyAuthentication yes/cPubkeyAuthentication yes' /etc/ssh/sshd_config && mkdir /run/sshd && echo "root:enlarge" | chpasswd
RUN . /opt/spack/share/spack/setup-env.sh && spack install gcc@9.3.0 
RUN . /opt/spack/share/spack/setup-env.sh && spack load gcc@9.3.0 && spack compiler find && spack install nccl@2.7.8-1%gcc@9.3.0
RUN cd /nest-simulator && git checkout v3.2 && cmake -Dwith-mpi=ON -DCMAKE_INSTALL_PREFIX:PATH=/nest-v3.2 /nest-simulator && make -j && make install
# CMD /bin/bash; sleep infinity
CMD /usr/sbin/sshd -D > ~/sshd.log 2> ~/sshd.err
