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
#include "optix/ModuleBuilder.hpp"


class PipelineBuilder{

    private:

        OptixPipelineCompileOptions pipelineCompileOptions_ = {};
        OptixPipelineLinkOptions pipelineLinkOptions_ = {};
        OptixPipeline pipeline_ = nullptr;

        char pipelineLog_[4096];
        size_t pipelineLogSize_ = sizeof(this->pipelineLog_);

    public:

        PipelineBuilder(){
            this->initialisePipelineCompileOptions() ;
            this->initialisePipelineLinkOptions() ;
        }

        OptixPipelineCompileOptions& pipelineCompileOptions(){
            return this->pipelineCompileOptions_ ;
        }

        OptixPipelineLinkOptions& pipelineLinkOptions(){
            return this->pipelineLinkOptions_ ;
        }

        OptixPipeline& pipeline(){
            return this->pipeline_ ;
        }

        char* pipelineLog(){
            return this->pipelineLog_ ;
        }

        size_t& pipelineLogSize(){
            return this->pipelineLogSize_ ;
        }

        void initialisePipelineCompileOptions() ;
        void initialisePipelineLinkOptions() ;

        OptixResult createOptixPipeline(OptixDeviceContext context, OptixProgramGroup programGroups[]) ;
        OptixResult setOptixPipelineStackSize() ;


} ;