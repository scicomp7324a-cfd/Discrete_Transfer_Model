#include <iostream>
#include <vector>
#include <array>

#include <cuda.h>
#include <cuda_runtime.h>

#include <optix.h>
#include <optix_stubs.h>
#include <optix_device.h>
#include <stdexcept>

#include "core/Triangle.hpp"
#include "optix/GasBuilder.hpp"
#include "optix/CudaOptixHelpers.hpp"

void GasBuilder::buildTriangleGas(OptixDeviceContext context, cudaStream_t stream, const std::vector<Vec3>& vertices, const std::vector<Tri>& indices){

    if (vertices.empty()){
        throw std::runtime_error("buildTriangleGas: vertices are empty");
    }
    
    if (indices.empty()){
        throw std::runtime_error("buildTriangleGas: indices are empty");
    }

    this->numVertices() = static_cast<unsigned int>(vertices.size()); 
    this->numTriangles() = static_cast<unsigned int>(indices.size());
    
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&this->d_vertices()), vertices.size()*sizeof(Vec3))) ;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&this->d_indices()), indices.size()*sizeof(Tri))) ;
    
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(this->d_vertices()), vertices.data(), vertices.size() * sizeof(Vec3),cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(this->d_indices()), indices.data(), indices.size() * sizeof(Tri),cudaMemcpyHostToDevice));

    OptixBuildInput buildInput = {};
    buildInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

    CUdeviceptr vertexBuffers[] = { this->d_vertices() };
    uint32_t triangleInputFlags[1] = { OPTIX_GEOMETRY_FLAG_NONE };

    buildInput.triangleArray.vertexBuffers       = vertexBuffers;
    buildInput.triangleArray.numVertices         = static_cast<unsigned int>(vertices.size());
    buildInput.triangleArray.vertexFormat        = OPTIX_VERTEX_FORMAT_FLOAT3;
    buildInput.triangleArray.vertexStrideInBytes = sizeof(Vec3);

    buildInput.triangleArray.indexBuffer         = this->d_indices() ;
    buildInput.triangleArray.numIndexTriplets    = static_cast<unsigned int>(indices.size());
    buildInput.triangleArray.indexFormat         = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    buildInput.triangleArray.indexStrideInBytes  = sizeof(Tri);

    buildInput.triangleArray.flags               = triangleInputFlags;
    buildInput.triangleArray.numSbtRecords       = 1;

    OptixAccelBuildOptions accelOptions = {};
    accelOptions.buildFlags = OPTIX_BUILD_FLAG_NONE;
    accelOptions.operation  = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes gasSizes = {};
    OPTIX_CHECK(optixAccelComputeMemoryUsage(context, &accelOptions, &buildInput, 1, &gasSizes)) ; 

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&this->d_tempBuffer()), gasSizes.tempSizeInBytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&this->d_gasBuffer()),  gasSizes.outputSizeInBytes));

    OPTIX_CHECK(optixAccelBuild(
        context,
        stream,
        &accelOptions,
        &buildInput,
        1,
        this->d_tempBuffer(),
        gasSizes.tempSizeInBytes,
        this->d_gasBuffer(),
        gasSizes.outputSizeInBytes,
        &this->handle(),
        nullptr,
        0
    ));

    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(this->d_tempBuffer())));
    this->d_tempBuffer() = 0;

}