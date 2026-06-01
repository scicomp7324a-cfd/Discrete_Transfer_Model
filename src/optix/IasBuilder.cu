#include <iostream>
#include <vector>
#include <array>

#include <cuda.h>
#include <cuda_runtime.h>

#include <optix.h>
#include <optix_stubs.h>
#include <optix_device.h>
#include <cstring>
#include <stdexcept>

#include "optix/IasBuilder.hpp"
#include "optix/GasBuilder.hpp"
#include "optix/CudaOptixHelpers.hpp"


void IasBuilder::buildIas(OptixDeviceContext context, cudaStream_t stream, GasBuilder& gasA, GasBuilder& gasB){

    std::array<OptixInstance, 2> instances = {};

    OptixInstance instA = {};
    float transformA[12];
    setIdentityTransform(transformA);
    std::memcpy(instA.transform, transformA, sizeof(float) * 12);
    instA.instanceId = 0;
    instA.sbtOffset = 0;
    instA.visibilityMask = 255;
    instA.flags = OPTIX_INSTANCE_FLAG_NONE;
    instA.traversableHandle = gasA.handle();

    OptixInstance instB = {};
    float transformB[12];
    setIdentityTransform(transformB);
    std::memcpy(instB.transform, transformB, sizeof(float) * 12);
    instB.instanceId = 1;
    instB.sbtOffset = 0;
    instB.visibilityMask = 255;
    instB.flags = OPTIX_INSTANCE_FLAG_NONE;
    instB.traversableHandle = gasB.handle();

    instances[0] = instA;
    instances[1] = instB;


    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&this->d_instances()),
                        sizeof(OptixInstance) * instances.size()));

    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(this->d_instances()),
                        instances.data(),
                        sizeof(OptixInstance) * instances.size(),
                        cudaMemcpyHostToDevice));

    OptixBuildInput iasInput = {};
    iasInput.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
    iasInput.instanceArray.instances = this->d_instances();
    iasInput.instanceArray.numInstances = static_cast<unsigned int>(instances.size());

    OptixAccelBuildOptions iasOptions = {};
    iasOptions.buildFlags = OPTIX_BUILD_FLAG_NONE;
    iasOptions.operation  = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes iasSizes = {};

    OPTIX_CHECK(optixAccelComputeMemoryUsage(
        context,
        &iasOptions,
        &iasInput,
        1,
        &iasSizes
    ));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&this->d_tempBuffer()), iasSizes.tempSizeInBytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&this->d_iasBuffer()),  iasSizes.outputSizeInBytes));

    OPTIX_CHECK(optixAccelBuild(
        context,
        stream,
        &iasOptions,
        &iasInput,
        1,
        this->d_tempBuffer(),
        iasSizes.tempSizeInBytes,
        this->d_iasBuffer(),
        iasSizes.outputSizeInBytes,
        &this->handle(),
        nullptr,
        0
    ));

    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(this->d_tempBuffer())));
    this->d_tempBuffer() = 0;

    
}