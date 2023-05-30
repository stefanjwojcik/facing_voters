#!/bin/bash

sudo apt-get update

#JULIA 
wget -O julia.tar.gz https://julialang-s3.julialang.org/bin/linux/x64/1.8/julia-1.8.0-linux-x86_64.tar.gz
tar -xvzf julia.tar.gz
rm julia.tar.gz
sudo mv julia-1.8.0 /opt/
sudo ln -s /opt/julia-1.8.0/bin/julia /usr/local/bin/julia

# Install R-base
sudo apt-get -y install r-base

# Install Tensorflow dependencies
sudo apt-get -y install python3-dev python3-pip python3-venv
sudo apt-get -y install libblas-dev liblapack-dev libatlas-base-dev gfortran

# Install Tensorflow
pip3 install tensorflow

# Install Nvidia driver and nvidia-smi
sudo apt-get -y install nvidia-driver-470
sudo apt-get -y install nvidia-utils-470

# Generate SSH key
#ssh-keygen -t rsa -b 4096 -C "your_email@example.com"

# Reboot the server to activate the Nvidia driver
#sudo reboot
