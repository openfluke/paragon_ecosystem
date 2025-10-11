#!/usr/bin/env bash
set -euo pipefail

echo "🧩 Starting environment setup for Paragon..."

# --- Detect package manager ---
if command -v dnf &>/dev/null; then
    PM="dnf"
    DISTRO="fedora"
elif command -v apt &>/dev/null; then
    PM="apt"
    DISTRO="ubuntu"
else
    echo "❌ Unsupported distro (needs dnf or apt)."
    exit 1
fi
echo "📦 Detected distro: $DISTRO"

# --- Python & pip ---
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

# --- Latest Go ---
echo "🦫 Installing latest Go..."
if command -v go &>/dev/null; then
    echo "✅ Go already installed: $(go version)"
else
    GOURL=$(curl -fsSL https://go.dev/VERSION?m=text | head -n1)
    GOVER=${GOURL/go/}
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

# --- .NET SDK 9.0 ---
echo "🛠️ Installing .NET SDK 9.0..."
if [ "$PM" = "dnf" ]; then
    sudo dnf install -y dotnet-sdk-9.0
else
    sudo apt update -y
    sudo apt install -y dotnet-sdk-9.0 || {
        echo "⚙️ Adding Microsoft package source for .NET 9.0..."
        wget https://packages.microsoft.com/config/ubuntu/24.04/packages-microsoft-prod.deb -O packages-microsoft-prod.deb
        sudo dpkg -i packages-microsoft-prod.deb
        sudo apt update -y
        sudo apt install -y dotnet-sdk-9.0
    }
fi

# --- Bun ---
echo "🌐 Installing Bun..."
if ! command -v bun &>/dev/null; then
    curl -fsSL https://bun.sh/install | bash
    export BUN_INSTALL="$HOME/.bun"
    export PATH="$BUN_INSTALL/bin:$PATH"
else
    echo "✅ Bun already installed."
fi

# --- Jupyter Notebook ---
echo "📓 Installing Jupyter Notebook..."
if ! command -v jupyter &>/dev/null; then
    python3 -m pip install --upgrade pip
    python3 -m pip install notebook jupyterlab
else
    echo "✅ Jupyter already installed."
fi

# --- Paragon Python package ---
echo "📦 Installing paragon-py..."
python3 -m pip install --upgrade pip
python3 -m pip install paragon-py

# --- Verify installs ---
echo
echo "✅ Versions check:"
python3 --version
pip3 --version
go version || echo "⚠️ Go install not verified"
dotnet --version || echo "⚠️ .NET install not verified"
bun --version || echo "⚠️ Bun install not verified"
jupyter --version || echo "⚠️ Jupyter install not verified"

echo
echo "🎉 Setup complete!"
echo "You can now run:"
echo "   python3 -m paragon.demo"
echo "   bun run your_script.ts"
echo "   dotnet new web -o paragon-web && cd paragon-web && dotnet run"
echo "   go run your_app.go"
echo "   jupyter notebook"
