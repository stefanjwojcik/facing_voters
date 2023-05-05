#!/bin/bash

# Add Julia PPA repository and install Julia
sudo add-apt-repository -y ppa:staticfloat/juliareleases
sudo apt-get update
sudo apt-get -y install julia=1.8.0

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
ssh-keygen -t rsa -b 4096 -C "your_email@example.com"

# Reboot the server to activate the Nvidia driver
sudo reboot
