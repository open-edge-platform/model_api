#!/bin/bash

# Added required keys / do the update
wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB

apt-key add GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB

echo "deb https://apt.repos.intel.com/openvino/2025 ubuntu22 main" | sudo tee /etc/apt/sources.list.d/intel-openvino-2025.list

apt update

#Install OpenCV
apt-get install libopencv-dev

# Install OpenVINO
sudo apt install openvino-2025.0.0
