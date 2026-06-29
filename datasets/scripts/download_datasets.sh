#!/usr/bin/env bash
# =============================================================================
# Dataset Download Script
# =============================================================================
# Downloads EuRoC MAV, TUM-VI, and KITTI Odometry datasets.
#
# Usage:
#   bash datasets/scripts/download_datasets.sh [dataset] [target_dir]
#
#   dataset    : euroc | tumvi | kitti | all  (default: all)
#   target_dir : where to save files         (default: datasets/data/)
#
# Examples:
#   bash datasets/scripts/download_datasets.sh euroc datasets/data/
#   bash datasets/scripts/download_datasets.sh kitti datasets/data/
#   bash datasets/scripts/download_datasets.sh all datasets/data/
#
# Disk space required:
#   EuRoC (3 sequences, no images):  ~450 MB
#   TUM-VI (2 sequences, no images): ~300 MB
#   KITTI  (poses + times only):     ~5 MB
#
# After downloading, update DATASET_PATHS in benchmark/run_phase6_real_datasets.py
# to point to the correct directories.
# =============================================================================

set -euo pipefail

DATASET="${1:-all}"
TARGET="${2:-$(dirname "$(dirname "$0")")/data}"
mkdir -p "$TARGET"

echo "============================================================"
echo "  EKF/UKF Phase 6 — Dataset Download"
echo "  Target: $TARGET"
echo "============================================================"

# =============================================================================
# EuRoC MAV Dataset
# https://rpg.ifi.uzh.ch/research_vo.html
# Burri et al., IJRR 2016
# =============================================================================
download_euroc() {
    echo ""
    echo "── EuRoC MAV Dataset ──────────────────────────────────────────"
    mkdir -p "$TARGET/EuRoC"

    BASE="http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset"

    declare -A SEQUENCES=(
        ["MH_01_easy"]="machine_hall/MH_01_easy/MH_01_easy.zip"
        ["V1_01_easy"]="vicon_room1/V1_01_easy/V1_01_easy.zip"
        ["V2_02_medium"]="vicon_room2/V2_02_medium/V2_02_medium.zip"
    )

    for SEQ in "${!SEQUENCES[@]}"; do
        URL="$BASE/${SEQUENCES[$SEQ]}"
        OUT="$TARGET/EuRoC/${SEQ}.zip"
        if [ -d "$TARGET/EuRoC/$SEQ" ]; then
            echo "  [SKIP] $SEQ already exists"
        else
            echo "  [GET]  $SEQ"
            echo "         URL: $URL"
            wget -q --show-progress -O "$OUT" "$URL" || \
                curl -L -o "$OUT" "$URL"
            echo "  [UNZIP] $SEQ"
            unzip -q "$OUT" -d "$TARGET/EuRoC/$SEQ"
            rm -f "$OUT"
            echo "  [DONE]  $TARGET/EuRoC/$SEQ"
        fi
    done
}

# =============================================================================
# TUM-VI Dataset (EuRoC-exported format)
# https://vision.in.tum.de/data/datasets/visual-inertial-dataset
# Schubert et al., IROS 2018
# =============================================================================
download_tumvi() {
    echo ""
    echo "── TUM-VI Dataset ─────────────────────────────────────────────"
    mkdir -p "$TARGET/TUM-VI"

    BASE="https://vision.in.tum.de/tumvi/exported/euroc/512_16"

    declare -A SEQUENCES=(
        ["room1"]="dataset-room1_512_16.tar.gz"
        ["corridor1"]="dataset-corridor1_512_16.tar.gz"
    )

    for SEQ in "${!SEQUENCES[@]}"; do
        URL="$BASE/${SEQUENCES[$SEQ]}"
        OUT="$TARGET/TUM-VI/${SEQUENCES[$SEQ]}"
        if [ -d "$TARGET/TUM-VI/$SEQ" ]; then
            echo "  [SKIP] $SEQ already exists"
        else
            echo "  [GET]  $SEQ"
            echo "         URL: $URL"
            wget -q --show-progress -O "$OUT" "$URL" || \
                curl -L -o "$OUT" "$URL"
            echo "  [UNTAR] $SEQ"
            mkdir -p "$TARGET/TUM-VI/$SEQ"
            tar -xzf "$OUT" -C "$TARGET/TUM-VI/$SEQ" --strip-components=1
            rm -f "$OUT"
            echo "  [DONE]  $TARGET/TUM-VI/$SEQ"
        fi
    done
}

# =============================================================================
# KITTI Odometry Dataset (poses + timestamps only)
# http://www.cvlibs.net/datasets/kitti/eval_odometry.php
# Geiger et al., CVPR 2012
# NOTE: Only poses and timestamps are downloaded (no images = fast).
# =============================================================================
download_kitti() {
    echo ""
    echo "── KITTI Odometry Dataset ─────────────────────────────────────"
    mkdir -p "$TARGET/KITTI/poses"

    # Ground truth poses (~5 MB)
    POSES_URL="https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_poses.zip"
    if [ -f "$TARGET/KITTI/poses/00.txt" ]; then
        echo "  [SKIP] poses already exist"
    else
        echo "  [GET]  poses (ground truth, ~5 MB)"
        echo "         URL: $POSES_URL"
        wget -q --show-progress -O "$TARGET/KITTI/poses.zip" "$POSES_URL" || \
            curl -L -o "$TARGET/KITTI/poses.zip" "$POSES_URL"
        echo "  [UNZIP] poses"
        unzip -q "$TARGET/KITTI/poses.zip" -d "$TARGET/KITTI/"
        rm -f "$TARGET/KITTI/poses.zip"
        echo "  [DONE]  $TARGET/KITTI/poses/"
    fi

    # Timestamps (~50 KB per sequence, sequences 00 and 05)
    for SEQ in 00 05; do
        mkdir -p "$TARGET/KITTI/sequences/$SEQ"
        TIMES_URL="https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_calib.zip"
        if [ -f "$TARGET/KITTI/sequences/$SEQ/times.txt" ]; then
            echo "  [SKIP] times.txt for sequence $SEQ already exists"
        else
            echo "  [NOTE] times.txt for sequence $SEQ should be in the calib zip"
            echo "         Manual download: http://www.cvlibs.net/datasets/kitti/eval_odometry.php"
            echo "         Or use the synthetic KITTI emulator (no download needed)"
        fi
    done
}

# =============================================================================
# Main
# =============================================================================
case "$DATASET" in
    euroc) download_euroc ;;
    tumvi) download_tumvi ;;
    kitti) download_kitti ;;
    all)
        download_euroc
        download_tumvi
        download_kitti
        ;;
    *)
        echo "ERROR: Unknown dataset '$DATASET'"
        echo "Usage: bash download_datasets.sh [euroc|tumvi|kitti|all] [target_dir]"
        exit 1
        ;;
esac

echo ""
echo "============================================================"
echo "  Download complete."
echo ""
echo "  Next step: update DATASET_PATHS in"
echo "  benchmark/run_phase6_real_datasets.py"
echo "  to point to: $TARGET"
echo "============================================================"
