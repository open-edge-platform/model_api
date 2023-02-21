#!/bin/bash

# Added required keys / do the update
wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB

apt-key add GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB

echo "deb https://apt.repos.intel.com/openvino/2022 focal main" | sudo tee /etc/apt/sources.list.d/intel-openvino-2022.list

apt update

#Install OpenCV
apt-get install libopencv-dev 

# Install OpenVINO
apt install openvino

# Install GTest for CPP tests
apt-get install libgtest-dev