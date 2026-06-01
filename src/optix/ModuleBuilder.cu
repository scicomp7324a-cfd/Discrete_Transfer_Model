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
#include "optix/PipelineBuilder.hpp"
#include "optix/ModuleBuilder.hpp"

using namespace std ;

void ModuleBuilder::setModuleCompileOptions(){

    this->moduleCompileOptions_.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    this->moduleCompileOptions_.optLevel         = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    this->moduleCompileOptions_.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_DEFAULT;

}

void ModuleBuilder::createModule(OptixDeviceContext context, OptixPipelineCompileOptions pipelineCompileOptions, string ptxCode){

    OptixResult res = optixModuleCreate(
        context,
        &this->optixModuleCompileOptions(),
        &pipelineCompileOptions,
        ptxCode.c_str(),
        ptxCode.size(),
        this->moduleLog(),
        &this->moduleLogSize(),
        &this->module()
    );

    std::cout << "Module creation log:\n" << this->moduleLog() << "\n";

    if (res != OPTIX_SUCCESS)
    {
        std::cerr << "optixModuleCreate failed\n";
        throw std::runtime_error("optixModuleCreate failed");
    }

    std::cout << "Module created successfully\n";
}
