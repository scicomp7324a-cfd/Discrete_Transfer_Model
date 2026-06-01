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
#include "optix/PipelineBuilder.hpp"



void PipelineBuilder::initialisePipelineCompileOptions(){
    this->pipelineCompileOptions_.usesMotionBlur        = 0;
    this->pipelineCompileOptions_.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
    this->pipelineCompileOptions_.numPayloadValues      = 1;
    this->pipelineCompileOptions_.numAttributeValues    = 2;
    this->pipelineCompileOptions_.exceptionFlags        = OPTIX_EXCEPTION_FLAG_NONE;
    this->pipelineCompileOptions_.pipelineLaunchParamsVariableName = "params";
}

void PipelineBuilder::initialisePipelineLinkOptions(){
    this->pipelineLinkOptions_.maxTraceDepth = 1;
}

OptixResult PipelineBuilder::createOptixPipeline(OptixDeviceContext context, OptixProgramGroup programGroups[]){
    
    OptixResult resPipeline = optixPipelineCreate(
        context,
        &this->pipelineCompileOptions(),
        &this->pipelineLinkOptions(),
        programGroups,
        3,
        this->pipelineLog(),
        &this->pipelineLogSize(),
        &this->pipeline()
    );

    return resPipeline ;
}

OptixResult PipelineBuilder::setOptixPipelineStackSize(){

    OptixResult resStack = optixPipelineSetStackSize(
    this->pipeline(),
    0,    // directCallableStackSizeFromTraversal
    0,    // directCallableStackSizeFromState
    1024, // continuationStackSize
    2     // maxTraversableGraphDepth
    );

    return resStack ;
}