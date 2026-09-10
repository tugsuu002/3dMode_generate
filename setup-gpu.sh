#!/usr/bin/env bash
# Docker-т GPU дамжуулах тохиргоо (nvidia-container-toolkit).
# Ажиллуулах:  sudo bash setup-gpu.sh
set -euo pipefail

[ "$(id -u)" -eq 0 ] || { echo "АЛДАА: sudo-гоор ажиллуулна уу →  sudo bash setup-gpu.sh"; exit 1; }

step(){ echo; echo "── $*"; }

step "1/5 NVIDIA-гийн gpg түлхүүр нэмж байна"
install -d -m 0755 /usr/share/keyrings
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
echo "   ok"

step "2/5 apt эх сурвалж нэмж байна"
curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  > /etc/apt/sources.list.d/nvidia-container-toolkit.list
echo "   ok: /etc/apt/sources.list.d/nvidia-container-toolkit.list"

step "3/5 apt-ийн жагсаалт шинэчилж байна"
apt-get update -o Dir::Etc::sourcelist=/etc/apt/sources.list.d/nvidia-container-toolkit.list \
               -o Dir::Etc::sourceparts=- -o APT::Get::List-Cleanup=0 >/dev/null
echo "   ok"

step "4/5 nvidia-container-toolkit суулгаж байна"
DEBIAN_FRONTEND=noninteractive apt-get install -y nvidia-container-toolkit
nvidia-ctk runtime configure --runtime=docker
echo "   ok"

step "5/5 docker-ийг дахин ачаалж байна"
systemctl restart docker
sleep 3
echo "   ok"

step "Шалгалт"
if docker run --rm --gpus all ubuntu:24.04 nvidia-smi -L; then
  echo
  echo "✓ БҮТЛЭЭ. Одоо дараагийн алхам:"
  echo "    docker pull colmap/colmap:latest"
else
  echo
  echo "✗ Шалгалт бүтсэнгүй. Дээрх алдааг харна уу."
  exit 1
fi
