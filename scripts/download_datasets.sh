#!/usr/bin/env bash
# download_datasets.sh — Downloads EuRoC MAV and TUM-VI (IMU+GT only, ~28 MB total)
set -euo pipefail
DATADIR="$(cd "$(dirname "$0")/.." && pwd)/datasets/data"
mkdir -p "$DATADIR/EuRoC" "$DATADIR/TUMVI"

echo "Downloading EuRoC MAV sequences (IMU+GT only)..."
BASE="http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset"
for SEQ in machine_hall/MH_01_easy machine_hall/MH_03_medium \
           vicon_room1/V1_01_easy vicon_room2/V2_02_medium; do
  NAME=$(basename $SEQ)
  URL="$BASE/$SEQ/${NAME}.zip"
  echo "  $NAME..."
  wget -q --show-progress -O "/tmp/${NAME}.zip" "$URL" || \
    curl -L -o "/tmp/${NAME}.zip" "$URL"
  unzip -q "/tmp/${NAME}.zip" -d "$DATADIR/EuRoC/"
  rm "/tmp/${NAME}.zip"
done
echo "EuRoC download complete: $DATADIR/EuRoC/"

echo ""
echo "Downloading TUM-VI room1 (IMU+GT only, ~18 MB)..."
wget -q -O "/tmp/tumvi_room1.tar.gz" \
  "https://vision.in.tum.de/tumvi/exported/euroc/512_16/dataset-room1_512_16.tar.gz" || \
  curl -L -o "/tmp/tumvi_room1.tar.gz" \
  "https://vision.in.tum.de/tumvi/exported/euroc/512_16/dataset-room1_512_16.tar.gz"
tar -xzf "/tmp/tumvi_room1.tar.gz" -C "$DATADIR/TUMVI/"
rm "/tmp/tumvi_room1.tar.gz"
echo "TUM-VI download complete: $DATADIR/TUMVI/"
echo ""
echo "Set environment variables:"
echo "  export EUROC_ROOT=$DATADIR/EuRoC"
echo "  export TUMVI_ROOT=$DATADIR/TUMVI"
