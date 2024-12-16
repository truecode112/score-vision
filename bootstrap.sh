#!/bin/bash

# install_prerequisites.sh
set -e  # Exit on any error

echo "Installing system prerequisites for sn44..."

# Update package list
sudo apt-get update

# Install Python 3.10 and related tools
sudo apt-get install -y \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    python3-pip \
    git \
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    curl \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libxml2-dev \
    libxmlsec1-dev \
    libffi-dev \
    liblzma-dev

# Install Node.js and npm
echo "Installing Node.js and npm..."
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt-get install -y nodejs

# Verify Node.js installation
node --version
npm --version

# Install PM2 globally
echo "Installing PM2..."
sudo npm install -g pm2

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Add uv to PATH for current session
export PATH="$HOME/.local/bin:$PATH"

# Add uv to PATH permanently
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc

# Source the updated PATH
source ~/.bashrc

echo "Prerequisites installation complete!"
echo "PM2 is now installed. You can manage processes using 'pm2' command."
echo "You can now proceed with the installation steps from README.md:"
