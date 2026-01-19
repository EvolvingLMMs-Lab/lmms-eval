#!/bin/bash

# 获取Ubuntu版本号，并提取版本编号（例如：20.04）
version=$(lsb_release -sr)

# 下载Microsoft软件包
sudo wget https://packages.microsoft.com/config/ubuntu/$version/packages-microsoft-prod.deb

# 安装下载的包
sudo dpkg -i packages-microsoft-prod.deb

# 更新软件源列表
sudo apt-get update

# 安装blobfuse2
sudo apt-get install blobfuse2

mkdir -p mycontainer