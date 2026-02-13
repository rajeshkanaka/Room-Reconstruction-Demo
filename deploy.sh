#!/bin/bash
# ================================================================
# Deploy to RunPod Pod — one command
# ================================================================
# Usage:
#   ./deploy.sh <IP> <PORT>
#   ./deploy.sh 69.30.85.149 22076
#
# Reads IP and PORT from the pod's "SSH over exposed TCP" section.
# ================================================================
set -e

if [ $# -lt 2 ]; then
    echo "Usage: ./deploy.sh <IP> <PORT>"
    echo "Example: ./deploy.sh 69.30.85.149 22076"
    exit 1
fi

IP="$1"
PORT="$2"
KEY="$HOME/.ssh/id_ed25519"
SSH="ssh -p $PORT -o StrictHostKeyChecking=no -o IdentitiesOnly=yes -i $KEY root@$IP"
SCP="scp -P $PORT -o StrictHostKeyChecking=no -o IdentitiesOnly=yes -i $KEY"
DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Deploying to root@$IP:$PORT"

# 1. Upload GCP credentials (if present locally)
GCP_KEY="$DIR/gcloud-sa.json"
if [ ! -f "$GCP_KEY" ]; then
    GCP_KEY="$HOME/gcloud-sa.json"
fi
if [ -f "$GCP_KEY" ]; then
    echo "[1/3] Uploading GCP credentials..."
    $SCP "$GCP_KEY" root@$IP:/workspace/gcloud-sa.json
else
    echo "[1/3] No gcloud-sa.json found, skipping (Gemini will be disabled)"
fi

# 2. Upload setup script
echo "[2/3] Uploading setup script..."
$SCP "$DIR/setup_runpod.sh" root@$IP:/workspace/setup_runpod.sh

# 3. Run setup
echo "[3/3] Running setup on pod (this takes a few minutes on first run)..."
$SSH "bash /workspace/setup_runpod.sh"
