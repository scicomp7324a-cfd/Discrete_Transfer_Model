#include <iostream>
#include <vector>
#include <array>

#include <cuda.h>
#include <cuda_runtime.h>

#include <optix.h>
#include <optix_stubs.h>
#include <optix_device.h>

#include "optix/ProgramGroupBuilder.hpp"
#include "optix/SbtBuilder.hpp"


void SbtBuilder::initialiseSbt(OptixProgramGroup& raygenPG, OptixProgramGroup& missPG, OptixProgramGroup& hitPG){

    using RaygenRecord = SbtRecord<int>;
    using MissRecord   = SbtRecord<int>;
    using HitRecord    = SbtRecord<int>;

    RaygenRecord rgRecord = {};
    MissRecord   msRecord = {};
    HitRecord    hgRecord = {};

    optixSbtRecordPackHeader(raygenPG, &rgRecord);
    optixSbtRecordPackHeader(missPG,   &msRecord);
    optixSbtRecordPackHeader(hitPG,    &hgRecord);

    CUdeviceptr d_rgRecord = 0;
    CUdeviceptr d_msRecord = 0;
    CUdeviceptr d_hgRecord = 0;

    cudaMalloc(reinterpret_cast<void**>(&d_rgRecord), sizeof(RaygenRecord));
    cudaMalloc(reinterpret_cast<void**>(&d_msRecord), sizeof(MissRecord));
    cudaMalloc(reinterpret_cast<void**>(&d_hgRecord), sizeof(HitRecord));

    cudaMemcpy(reinterpret_cast<void*>(d_rgRecord), &rgRecord, sizeof(RaygenRecord), cudaMemcpyHostToDevice);
    cudaMemcpy(reinterpret_cast<void*>(d_msRecord), &msRecord, sizeof(MissRecord),   cudaMemcpyHostToDevice);
    cudaMemcpy(reinterpret_cast<void*>(d_hgRecord), &hgRecord, sizeof(HitRecord),    cudaMemcpyHostToDevice);

    this->sbt().raygenRecord                = d_rgRecord;
    this->sbt().missRecordBase              = d_msRecord;
    this->sbt().missRecordStrideInBytes     = sizeof(MissRecord);
    this->sbt().missRecordCount             = 1;
    this->sbt().hitgroupRecordBase          = d_hgRecord;
    this->sbt().hitgroupRecordStrideInBytes = sizeof(HitRecord);
    this->sbt().hitgroupRecordCount         = 1;

}