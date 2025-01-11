FROM ubuntu:22.04

# To Install NVIDIA For ubuntu 22.04
#wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
#sudo dpkg -i cuda-keyring_1.0-1_all.deb
#sudo apt-get update
#sudo apt-get -y install cuda

RUN apt-get update
RUN apt-get -y install wget

#JULIA 
RUN wget -O julia.tar.gz https://julialang-s3.julialang.org/bin/linux/x64/1.8/julia-1.8.5-linux-x86_64.tar.gz
RUN tar -xvzf julia.tar.gz
RUN rm julia.tar.gz
RUN mv julia-1.8.5 /opt/
RUN ln -s /opt/julia-1.8.5/bin/julia /usr/local/bin/julia

## Install Pluto on Julia 
RUN julia -e 'using Pkg; Pkg.add("Pluto")'

# Install R-base - consider RUN apt-get install -y r-base=4.1.2 r-base-dev=4.1.2
RUN apt-get -y install r-base
RUN apt-get install r-base-dev=4.1.2

# Install Tensorflow dependencies
RUN apt-get -y install python3-dev python3-pip python3-venv
RUN apt-get -y install libblas-dev liblapack-dev libatlas-base-dev gfortran

# Install Tensorflow
RUN cd src/python && pip install -r requirements.txt

# Install Nvidia driver and nvidia-smi
#sudo apt-get -y install nvidia-driver-470
#sudo apt-get -y install nvidia-utils-470

# Generate SSH key
#ssh-keygen -t rsa -b 4096 -C "your_email@example.com"

# Reboot the server to activate the Nvidia driver
#sudo reboot

# Installing python requirements
# pip freeze >> requirements.txt
