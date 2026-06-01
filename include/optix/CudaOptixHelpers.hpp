#pragma once

#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>
#include <optix.h>

#define CUDA_CHECK(call)                                                     \
do {                                                                         \
    cudaError_t rc = (call);                                                 \
    if (rc != cudaSuccess) {                                                 \
        std::cerr << "CUDA error: " << cudaGetErrorString(rc)                \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl;     \
        std::exit(1);                                                        \
    }                                                                        \
} while(0)

#define OPTIX_CHECK(call)                                                    \
do {                                                                         \
    OptixResult res = (call);                                                \
    if (res != OPTIX_SUCCESS) {                                              \
        std::cerr << "OptiX error: " << static_cast<int>(res)                \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl;     \
        std::exit(1);                                                        \
    }                                                                        \
} while(0)
