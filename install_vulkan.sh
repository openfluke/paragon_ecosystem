#!/usr/bin/env bash
set -euo pipefail

echo "🧠 Installing Vulkan SDK + drivers + validation layers..."

if command -v dnf &>/dev/null; then
  echo "📦 Detected Fedora"
  sudo dnf install -y \
    vulkan \
    vulkan-tools \
    vulkan-loader-devel \
    mesa-vulkan-drivers \
    vulkan-validation-layers-devel \
    spirv-tools \
    glslang

elif command -v apt &>/dev/null; then
  echo "📦 Detected Ubuntu/Debian"
  sudo apt update -y
  sudo apt install -y \
    mesa-vulkan-drivers \
    vulkan-tools \
    libvulkan1 \
    vulkan-validationlayers \
    vulkan-validationlayers-dev \
    spirv-tools \
    glslang-tools

else
  echo "❌ Unsupported distro. Please install Vulkan SDK manually from https://vulkan.lunarg.com/sdk/home"
  exit 1
fi

echo
echo "✅ Vulkan installed successfully!"
echo "Run to verify:"
echo "   vulkaninfo | less"
echo "   vkcube"
