#!/usr/bin/env bash
set -euo pipefail

echo "🧩 Starting environment setup for Paragon..."

# -------------------------------
# Detect package manager / distro
# -------------------------------
if command -v dnf &>/dev/null; then
  PM="dnf"; DISTRO="fedora"
elif command -v apt &>/dev/null; then
  PM="apt"; DISTRO="ubuntu"
else
  echo "❌ Unsupported distro (needs dnf or apt)."; exit 1
fi
echo "📦 Detected distro: $DISTRO"

append_once() {
  local line="$1" file="$2"
  grep -qxF "$line" "$file" 2>/dev/null || echo "$line" >> "$file"
}

# -------------------------------
# System compiler toolchain (GCC)
# -------------------------------
echo "🧱 Installing GCC toolchain..."
if [ "$PM" = "dnf" ]; then
  sudo dnf install -y gcc g++ make
else
  sudo apt update -y && sudo apt install -y build-essential
fi

# sanity check
if ! command -v gcc &>/dev/null; then
  echo "❌ gcc not found after install, check your PATH!"
  exit 1
else
  echo "✅ gcc detected at $(which gcc)"
fi

# persist CGO + compiler paths
append_once 'export CGO_ENABLED=1' "$HOME/.bashrc"
append_once 'export CC=/usr/bin/gcc' "$HOME/.bashrc"
append_once 'export CXX=/usr/bin/g++' "$HOME/.bashrc"
export CGO_ENABLED=1
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++

# -------------------------------
# Python + pip
# -------------------------------
if ! command -v python3 &>/dev/null; then
  echo "🐍 Installing Python 3..."
  [ "$PM" = "dnf" ] && sudo dnf install -y python3 || \
    (sudo apt update -y && sudo apt install -y python3)
else
  echo "✅ Python 3 already present."
fi

if ! command -v pip3 &>/dev/null; then
  echo "⚙️ Installing pip..."
  [ "$PM" = "dnf" ] && sudo dnf install -y python3-pip || sudo apt install -y python3-pip
else
  echo "✅ pip already installed."
fi

# -------------------------------
# Go
# -------------------------------
echo "🦫 Installing latest Go (if missing)..."
if command -v go &>/dev/null; then
  echo "✅ Go already installed: $(go version)"
else
  GOURL=$(curl -fsSL https://go.dev/VERSION?m=text | head -n1)
  ARCH=$(uname -m)
  case "$ARCH" in
    x86_64) GOARCH="amd64" ;;
    aarch64|arm64) GOARCH="arm64" ;;
    *) echo "⚠️ Unsupported arch $ARCH"; exit 1 ;;
  esac
  wget "https://go.dev/dl/${GOURL}.linux-${GOARCH}.tar.gz" -O /tmp/go.tar.gz
  sudo rm -rf /usr/local/go
  sudo tar -C /usr/local -xzf /tmp/go.tar.gz
  echo "export PATH=/usr/local/go/bin:\$PATH" | sudo tee /etc/profile.d/go.sh >/dev/null
  export PATH=/usr/local/go/bin:$PATH
  echo "✅ Installed $(go version)"
fi

# -------------------------------
# .NET SDK 9.0
# -------------------------------
echo "🛠️ Installing .NET SDK 9.0..."
if [ "$PM" = "dnf" ]; then
  sudo dnf install -y dotnet-sdk-9.0 || true
else
  sudo apt update -y
  sudo apt install -y dotnet-sdk-9.0 || {
    echo "⚙️ Adding Microsoft repo..."
    wget -q https://packages.microsoft.com/config/ubuntu/24.04/packages-microsoft-prod.deb -O /tmp/packages-microsoft-prod.deb
    sudo dpkg -i /tmp/packages-microsoft-prod.deb
    sudo apt update -y && sudo apt install -y dotnet-sdk-9.0
  }
fi

# -------------------------------
# Bun
# -------------------------------
echo "🌐 Installing Bun..."
if ! command -v bun &>/dev/null; then
  curl -fsSL https://bun.sh/install | bash
  append_once 'export BUN_INSTALL="$HOME/.bun"' "$HOME/.bashrc"
  append_once 'export PATH="$BUN_INSTALL/bin:$PATH"' "$HOME/.bashrc"
  export BUN_INSTALL="$HOME/.bun"
  export PATH="$BUN_INSTALL/bin:$PATH"
else
  echo "✅ Bun already installed: $(bun --version)"
fi

# -------------------------------
# Node.js + npm
# -------------------------------
echo "🟩 Installing Node.js (LTS) + npm..."
if command -v node &>/dev/null; then
  echo "✅ Node already installed: $(node -v)"
else
  if [ "$PM" = "dnf" ]; then
    curl -fsSL https://rpm.nodesource.com/setup_lts.x | sudo bash -
    sudo dnf install -y nodejs
  else
    curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash -
    sudo apt install -y nodejs
  fi
  echo "✅ Installed Node $(node -v), npm $(npm -v)"
fi

NPM_PREFIX=$(npm config get prefix 2>/dev/null || echo "")
if [[ -n "$NPM_PREFIX" && "$NPM_PREFIX" != "/usr" && "$NPM_PREFIX" != "/usr/local" ]]; then
  append_once "export PATH=\"$NPM_PREFIX/bin:\$PATH\"" "$HOME/.bashrc"
  export PATH="$NPM_PREFIX/bin:$PATH"
fi

# -------------------------------
# Ionic CLI
# -------------------------------
echo "⚡ Installing Ionic CLI..."
if ! command -v ionic &>/dev/null; then
  sudo npm i -g @ionic/cli
else
  echo "✅ Ionic already installed: $(ionic --version)"
fi

# -------------------------------
# Jupyter
# -------------------------------
echo "📓 Installing Jupyter (Notebook + Lab)..."
if ! command -v jupyter &>/dev/null; then
  python3 -m pip install --upgrade pip
  python3 -m pip install notebook jupyterlab
else
  echo "✅ Jupyter already installed."
fi

# -------------------------------
# paragon-py
# -------------------------------
echo "📦 Installing paragon-py..."
python3 -m pip install -U pip paragon-py

# -------------------------------
# Summary
# -------------------------------
echo
echo "✅ Versions check:"
python3 --version || true
pip3 --version || true
go version || true
dotnet --version || true
gcc --version || true
g++ --version || true
node -v || true
npm -v || true
ionic --version || true
bun --version || true
jupyter --version || true

echo
echo "🎉 Setup complete!"
echo "Restart your terminal or run:"
echo "   source ~/.bashrc"
echo
echo "You can now compile native Go projects with CGO + Vulkan."
