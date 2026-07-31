#!/usr/bin/env bash

# Resolve project & script paths dynamically
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -f "$SCRIPT_DIR/pyproject.toml" ]; then
    PROJECT_DIR="$SCRIPT_DIR"
elif [ -f "$SCRIPT_DIR/../pyproject.toml" ]; then
    PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
else
    PROJECT_DIR="$(pwd)"
fi

LAUNCHER_SCRIPT="$PROJECT_DIR/scripts/lanzar_ds_guardian.sh"
TARGET_DIR="${HOME}/.local/share/applications"
TARGET_DESKTOP="$TARGET_DIR/DS_Guardian.desktop"

mkdir -p "$TARGET_DIR"

cat <<EOF > "$TARGET_DESKTOP"
[Desktop Entry]
Version=1.0
Type=Application
Name=DS Guardian
Comment=AI Data Science Governance System & Data Quality Auditor
Exec=gnome-terminal -- /bin/bash -c "$LAUNCHER_SCRIPT; exec bash"
Icon=utilities-terminal
Terminal=true
Categories=Development;Science;
StartupNotify=true
EOF

chmod +x "$TARGET_DESKTOP"
chmod +x "$LAUNCHER_SCRIPT"

echo "============================================================"
echo " ✅ DS Guardian Desktop Shortcut Installed Successfully!"
echo " Location: $TARGET_DESKTOP"
echo " Exec Path: $LAUNCHER_SCRIPT"
echo "============================================================"
