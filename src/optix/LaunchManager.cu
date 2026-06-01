#include <iostream>
#include <vector>
#include <array>

#include <cuda.h>
#include <cuda_runtime.h>

#include <optix.h>
#include <optix_stubs.h>
#include <optix_device.h>

#include "optix/LaunchManager.hpp"
#include "radiation/DataBuilder.hpp"
#include "core/RayGeometry.hpp"
#include "core/TriangleTraceHistory.hpp"
#include "core/RayParams.hpp"
#include "optix/IasBuilder.hpp"
#include "optix/PipelineBuilder.hpp"
#include "optix/SbtBuilder.hpp"


void LaunchManager::initialiseDeviceVariables(){

    cudaMalloc(reinterpret_cast<void**>(&this->d_info()), this->numRays()*sizeof(RayGeometry));
    cudaMalloc(reinterpret_cast<void**>(&this->d_triangleInfo()), this->numTotalTriangles()*sizeof(TriangleTraceHistory));
    cudaMalloc(reinterpret_cast<void**>(&this->d_rayOrigins()), this->numRays()*sizeof(float3)) ;
    cudaMalloc(reinterpret_cast<void**>(&this->d_rayDirections()), this->numRays()*sizeof(float3)) ;
    cudaMalloc(reinterpret_cast<void**>(&this->d_originTriangleIds()), this->numRays()*sizeof(int)) ;
    cudaMalloc(reinterpret_cast<void**>(&this->d_originSurfaceIds()), this->numRays()*sizeof(int));
    cudaMalloc(reinterpret_cast<void**>(&this->d_hitRecords()), this->numRays()*sizeof(HitRecord));
    cudaMalloc(reinterpret_cast<void**>(&this->d_originGlobalTriangleIds()), this->numRays()*sizeof(int));
    cudaMalloc(reinterpret_cast<void**>(&this->d_surfaceTriangleOffsets()), this->h_surfaceTriangleOffsets_.size() * sizeof(int));

    cudaMemset(reinterpret_cast<void*>(this->d_info()), 0, this->numRays()*sizeof(RayGeometry));
    cudaMemset(reinterpret_cast<void*>(this->d_rayOrigins()), 0, this->numRays()*sizeof(float3)) ;
    cudaMemset(reinterpret_cast<void*>(this->d_rayDirections()), 0, this->numRays()*sizeof(float3)) ;
    cudaMemset(reinterpret_cast<void*>(this->d_originTriangleIds()), 0, this->numRays()*sizeof(int)) ;
    cudaMemset(reinterpret_cast<void*>(this->d_originSurfaceIds()), 0, this->numRays()* sizeof(int));
    cudaMemset(reinterpret_cast<void*>(this->d_hitRecords()), 0xff, this->numRays()*sizeof(HitRecord));

    cudaMemcpy(reinterpret_cast<void*>(this->d_rayOrigins()), this->h_rayOrigins().data(), this->numRays()*sizeof(float3), cudaMemcpyHostToDevice) ;
    cudaMemcpy(reinterpret_cast<void*>(this->d_rayDirections()), this->h_rayDirections().data(), this->numRays()*sizeof(float3), cudaMemcpyHostToDevice) ;
    cudaMemcpy(reinterpret_cast<void*>(this->d_originTriangleIds()), this->h_originTriangleIds().data(), this->numRays()*sizeof(int), cudaMemcpyHostToDevice) ;
    cudaMemcpy(reinterpret_cast<void*>(this->d_originSurfaceIds()), this->h_originSurfaceIds().data(), this->numRays()*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(reinterpret_cast<void*>(this->d_triangleInfo()), this->h_triangleInfo().data(), this->numTotalTriangles()*sizeof(TriangleTraceHistory),cudaMemcpyHostToDevice);
    cudaMemcpy(reinterpret_cast<void*>(this->d_originGlobalTriangleIds()), this->h_originGlobalTriangleIds().data(), this->numRays() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(reinterpret_cast<void*>(this->d_surfaceTriangleOffsets()), this->h_surfaceTriangleOffsets_.data(), this->h_surfaceTriangleOffsets_.size() * sizeof(int), cudaMemcpyHostToDevice);
}

void LaunchManager::initialiseHostParameters(DataBuilder& dataBuilder, IasBuilder& ias){

    this->h_params().handle = ias.handle() ;
    this->h_params().out = reinterpret_cast<RayGeometry*>(this->d_info()) ;
    this->h_params().rayOrigins = reinterpret_cast<float3*>(this->d_rayOrigins()) ;
    this->h_params().rayDirections = reinterpret_cast<float3*>(this->d_rayDirections()) ;
    this->h_params().originTriangleIds = reinterpret_cast<int*>(this->d_originTriangleIds()) ;
    this->h_params().originSurfaceIds  = reinterpret_cast<int*>(this->d_originSurfaceIds()) ;
    this->h_params().triangleHistory = reinterpret_cast<TriangleTraceHistory*>(this->d_triangleInfo()) ;
    this->h_params().hitRecords = reinterpret_cast<HitRecord*>(this->d_hitRecords()) ;
    this->h_params().numRays = this->numRays() ;
    this->h_params().numTrianglesA = dataBuilder.indicesA().size() ;
    this->h_params().numTrianglesB = dataBuilder.indicesB().size() ;
    this->h_params().numTotalTriangles = this->numTotalTriangles() ;
    this->h_params().originGlobalTriangleIds = reinterpret_cast<int*>(this->d_originGlobalTriangleIds());
    this->h_params().surfaceTriangleOffsets = reinterpret_cast<int*>(this->d_surfaceTriangleOffsets());

}

void LaunchManager::initialiseDeviceParameters(){

    cudaMalloc(reinterpret_cast<void**>(&this->d_params()), sizeof(Params));
    cudaMemcpy(reinterpret_cast<void*>(this->d_params()), &this->h_params(), sizeof(Params), cudaMemcpyHostToDevice);

}

OptixResult LaunchManager::LaunchOptixSimulation(PipelineBuilder& pipelineBuilder, SbtBuilder& sbtBuilder, cudaStream_t& stream){

    OptixResult resLaunch = optixLaunch(
    pipelineBuilder.pipeline(),
    stream,
    this->d_params(),
    sizeof(Params),
    &sbtBuilder.sbt(),
    this->numRays(), 1, 1
    );

    if (resLaunch != OPTIX_SUCCESS)
    {
        cerr << "optixLaunch failed\n";
        throw std::runtime_error("optixLaunch failed\n") ;
    }
    
    cudaStreamSynchronize(stream);
    //cudaMemcpy(this->h_info().data(), reinterpret_cast<void*>(this->d_info()), this->numRays()*sizeof(RayGeometry), cudaMemcpyDeviceToHost);
    //cudaMemcpy(this->h_hitRecords().data(), reinterpret_cast<void*>(this->d_hitRecords()), this->numRays()*sizeof(HitRecord), cudaMemcpyDeviceToHost);
    
    return resLaunch ;
}


