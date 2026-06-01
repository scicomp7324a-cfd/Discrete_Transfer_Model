# pragma once 

#include <iostream>
#include <vector>
#include <array>

#include <cuda.h>
#include <cuda_runtime.h>

#include <optix.h>
#include <optix_stubs.h>
#include <optix_device.h>

#include "core/Triangle.hpp"

using namespace std ;

class ProgramGroupBuilder{

    private:

        OptixProgramGroupOptions programGroupOptions_ = {};

        OptixProgramGroup raygenPG_ = nullptr;
        OptixProgramGroup missPG_   = nullptr;
        OptixProgramGroup hitPG_    = nullptr;

        char log_[4096];
        size_t logSize_ = sizeof(log_) ;

        OptixProgramGroupDesc raygenDesc_ = {};
        OptixProgramGroupDesc missDesc_ = {};
        OptixProgramGroupDesc hitDesc_ = {};


    public: 

        ProgramGroupBuilder(OptixDeviceContext context, OptixModule module){

            this->setRaygenDesc(module) ;
            this->createResRaygen(context) ;

            this->setHitDesc(module) ;
            this->createResHit(context) ;

            this->setMissDesc(module) ;
            this->createResMiss(context) ;

            std::cout << "All program groups created successfully\n";
        }

        OptixProgramGroupOptions& programGroupOptions(){
            return this->programGroupOptions_ ;
        }

        OptixProgramGroup& raygenPG(){
            return this->raygenPG_ ;
        }

        OptixProgramGroup& missPG(){
            return this->missPG_ ;
        }

        OptixProgramGroup& hitPG(){
            return this->hitPG_ ;
        }

        char* pgLog(){
        return this->log_ ;
        }

        size_t& pgLogSize(){
            return this->logSize_ ;
        }

        OptixProgramGroupDesc& raygenDesc(){
            return this->raygenDesc_ ;
        }

        OptixProgramGroupDesc& missDesc(){
            return this->missDesc_ ;
        }

        OptixProgramGroupDesc& hitDesc(){
            return this->hitDesc_ ;
        }

        void setRaygenDesc(OptixModule module); 
        void setMissDesc(OptixModule module);
        void setHitDesc(OptixModule module);

        void createResRaygen(OptixDeviceContext context);
        void createResHit(OptixDeviceContext context) ;
        void createResMiss(OptixDeviceContext context) ;


} ;