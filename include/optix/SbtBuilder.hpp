# pragma once 

#include <iostream>
#include <vector>
#include <array>

#include <cuda.h>
#include <cuda_runtime.h>

#include <optix.h>
#include <optix_stubs.h>
#include <optix_device.h>

template<typename T>
struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) SbtRecord
{
    char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

class SbtBuilder{

    private:

        OptixShaderBindingTable sbt_ = {};

    public:
        SbtBuilder(OptixProgramGroup& raygenPG, OptixProgramGroup& missPG, OptixProgramGroup& hitPG){
            this->initialiseSbt(raygenPG, missPG, hitPG) ;
        }

        OptixShaderBindingTable& sbt(){ return this->sbt_ ; }
        void initialiseSbt(OptixProgramGroup& raygenPG, OptixProgramGroup& missPG, OptixProgramGroup& hitPG) ;
} ;
