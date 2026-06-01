#!/usr/bin/env bash
set -e
rm -rf build

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"

export CUDA_HOME=/lsc/opt/cuda-12.5
export CUDA_PATH=/lsc/opt/cuda-12.5
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

echo "Using nvcc:"
which nvcc
nvcc --version

if [ -f "$BUILD_DIR/CMakeCache.txt" ]; then
    CACHE_SOURCE_DIR=$(grep '^CMAKE_HOME_DIRECTORY:INTERNAL=' "$BUILD_DIR/CMakeCache.txt" | cut -d= -f2)

    if [ "$CACHE_SOURCE_DIR" != "$SCRIPT_DIR" ]; then
        rm -rf "$BUILD_DIR"
    fi
fi

cmake -S "$SCRIPT_DIR" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_COMPILER=/lsc/opt/cuda-12.5/bin/nvcc \
    -DCUDAToolkit_ROOT=/lsc/opt/cuda-12.5 \
    -DCMAKE_CUDA_ARCHITECTURES=80-real

cmake --build "$BUILD_DIR" -j

cp "$BUILD_DIR/devicePrograms.ptx" "$SCRIPT_DIR/devicePrograms.ptx"

cd "$SCRIPT_DIR"
"$BUILD_DIR/dtm_app"