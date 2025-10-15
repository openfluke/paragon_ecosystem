#!/usr/bin/env bash
set -euo pipefail

echo "🧩 Starting environment setup for Paragon..."

# -------------------------------
# Detect package manager / distro
# -------------------------------
if command -v dnf &>/dev/null; then
  PM="dnf";   DISTRO="fedora"
elif command -v apt &>/dev/null; then
  PM="apt";   DISTRO="ubuntu"
else
  echo "❌ Unsupported distro (needs dnf or apt)."; exit 1
fi
echo "📦 Detected distro: $DISTRO"

# -------------------------------
# Helpers
# -------------------------------
append_once() {
  local line="$1" file="$2"
  grep -qxF "$line" "$file" 2>/dev/null || echo "$line" >> "$file"
}

# -------------------------------
# Python + pip
# -------------------------------
if ! command -v python3 &>/dev/null; then
  echo "🐍 Installing Python 3..."
  if [ "$PM" = "dnf" ]; then
    sudo dnf install -y python3
  else
    sudo apt update -y && sudo apt install -y python3
  fi
else
  echo "✅ Python 3 already present."
fi

if ! command -v pip3 &>/dev/null; then
  echo "⚙️ Installing pip..."
  if [ "$PM" = "dnf" ]; then
    sudo dnf install -y python3-pip
  else
    sudo apt install -y python3-pip
  fi
else
  echo "✅ pip already installed."
fi

# -------------------------------
# Latest Go (from go.dev tarball)
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
    echo "⚙️ Adding Microsoft package source for .NET 9.0..."
    wget -q https://packages.microsoft.com/config/ubuntu/24.04/packages-microsoft-prod.deb -O /tmp/packages-microsoft-prod.deb
    sudo dpkg -i /tmp/packages-microsoft-prod.deb
    sudo apt update -y
    sudo apt install -y dotnet-sdk-9.0
  }
fi

# -------------------------------
# Bun
# -------------------------------
echo "🌐 Installing Bun..."
if ! command -v bun &>/dev/null; then
  curl -fsSL https://bun.sh/install | bash
  # Persist Bun PATH for new terminals
  append_once 'export BUN_INSTALL="$HOME/.bun"' "$HOME/.bashrc"
  append_once 'export PATH="$BUN_INSTALL/bin:$PATH"' "$HOME/.bashrc"
  export BUN_INSTALL="$HOME/.bun"
  export PATH="$BUN_INSTALL/bin:$PATH"
else
  echo "✅ Bun already installed: $(bun --version)"
fi

# -------------------------------
# Node.js (LTS) + npm
# -------------------------------
echo "🟩 Installing Node.js (LTS) + npm..."
if command -v node &>/dev/null; then
  echo "✅ Node already installed: $(node -v)"
else
  if [ "$PM" = "dnf" ]; then
    # NodeSource for latest LTS on Fedora
    curl -fsSL https://rpm.nodesource.com/setup_lts.x | sudo bash -
    sudo dnf install -y nodejs
  else
    # NodeSource for Ubuntu/Debian
    curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash -
    sudo apt install -y nodejs
  fi
  echo "✅ Installed Node $(node -v), npm $(npm -v)"
fi

# Ensure global npm bin is in PATH if not installing with sudo
NPM_PREFIX=$(npm config get prefix 2>/dev/null || echo "")
if [[ -n "$NPM_PREFIX" && "$NPM_PREFIX" != "/usr" && "$NPM_PREFIX" != "/usr/local" ]]; then
  append_once "export PATH=\"$NPM_PREFIX/bin:\$PATH\"" "$HOME/.bashrc"
  export PATH="$NPM_PREFIX/bin:$PATH"
fi

# -------------------------------
# Ionic CLI (global)
# -------------------------------
echo "⚡ Installing Ionic CLI..."
if ! command -v ionic &>/dev/null; then
  # Use sudo to drop binaries into /usr/bin so new terminals see it immediately
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
python3 -m pip install --upgrade pip
python3 -m pip install -U paragon-py

# -------------------------------
# Versions check
# -------------------------------
echo
echo "✅ Versions check:"
python3 --version || true
pip3 --version || true
go version || echo "⚠️ Go install not verified"
dotnet --version || echo "⚠️ .NET install not verified"
node -v || echo "⚠️ Node install not verified"
npm -v || echo "⚠️ npm install not verified"
ionic --version || echo "⚠️ Ionic install not verified"
bun --version || echo "⚠️ Bun install not verified"
jupyter --version || echo "⚠️ Jupyter install not verified"

echo
echo "🎉 Setup complete!"
echo "Open a NEW terminal (or 'source ~/.bashrc') and you can run:"
echo "   python3 -m paragon.demo"
echo "   node -v && npm -v && ionic --version"
echo "   bun --version"
echo "   dotnet --version"
echo "   go version"
echo "   jupyter lab"
