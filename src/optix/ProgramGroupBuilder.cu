#include <iostream>
#include <vector>
#include <array>
#include <string>

#include <cuda.h>
#include <cuda_runtime.h>

#include <optix.h>
#include <optix_stubs.h>
#include <optix_device.h>

#include "core/Triangle.hpp"
#include "optix/ProgramGroupBuilder.hpp"

using namespace std ;

void ProgramGroupBuilder::setRaygenDesc(OptixModule module){
    this->raygenDesc().kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    this->raygenDesc().raygen.module = module ;
    this->raygenDesc().raygen.entryFunctionName = "__raygen__rg";
}

void ProgramGroupBuilder::setHitDesc(OptixModule module){
    this->hitDesc().kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    this->hitDesc().hitgroup.moduleCH = module ;
    this->hitDesc().hitgroup.entryFunctionNameCH = "__closesthit__ch";
    this->hitDesc().hitgroup.moduleAH = nullptr;
    this->hitDesc().hitgroup.entryFunctionNameAH = nullptr;
    this->hitDesc().hitgroup.moduleIS = nullptr;
    this->hitDesc().hitgroup.entryFunctionNameIS = nullptr;
}

void ProgramGroupBuilder::setMissDesc(OptixModule module){
    this->missDesc().kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    this->missDesc().miss.module = module;
    this->missDesc().miss.entryFunctionName = "__miss__ms";
}

void ProgramGroupBuilder::createResRaygen(OptixDeviceContext context){

    OptixResult resRaygen = optixProgramGroupCreate(
        context,
        &this->raygenDesc(),
        1,
        &this->programGroupOptions(),
        this->pgLog(),
        &this->pgLogSize(),
        &this->raygenPG()
    );

    std::cout << "Raygen PG log:\n" << this->pgLog() << "\n";
    if (resRaygen != OPTIX_SUCCESS)
    {
        std::cerr << "Raygen program group creation failed\n";
        throw std::runtime_error("Raygen program group creation failed");
    }

}

void ProgramGroupBuilder::createResHit(OptixDeviceContext context){

    OptixResult resHit = optixProgramGroupCreate(
        context,
        &this->hitDesc(),
        1,
        &this->programGroupOptions(),
        this->pgLog(),
        &this->pgLogSize(),
        &this->hitPG()
    );

    std::cout << "Hit PG log:\n" << this->pgLog() << "\n";
    if (resHit != OPTIX_SUCCESS)
    {
        std::cerr << "Hit program group creation failed\n";
        throw std::runtime_error("Hit program group creation failed");
    }
}

void ProgramGroupBuilder::createResMiss(OptixDeviceContext context){

    OptixResult resMiss = optixProgramGroupCreate(
        context,
        &this->missDesc(),
        1,
        &this->programGroupOptions(),
        this->pgLog(),
        &this->pgLogSize(),
        &this->missPG()
    );

    std::cout << "Miss PG log:\n" << this->pgLog() << "\n";
    if (resMiss != OPTIX_SUCCESS)
    {
        std::cerr << "Miss program group creation failed\n";
        throw std::runtime_error("Miss program group creation failed");
    }

}