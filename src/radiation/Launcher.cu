#include <iostream>
#include <vector>
#include <array>
#include <algorithm>
#include <cmath>

#include <cuda.h>
#include <cuda_runtime.h>

#include <optix.h>
#include <optix_stubs.h>
#include <optix_device.h>

#include <fstream>
#include <sstream>
#include <string>

#include <optix_function_table_definition.h>

#include "core/Triangle.hpp"
#include "core/RayGeometry.hpp"
#include "core/RayParams.hpp"
#include "core/PathConfig.hpp"

#include "optix/GasBuilder.hpp"
#include "optix/IasBuilder.hpp"
#include "optix/ModuleBuilder.hpp"
#include "optix/PipelineBuilder.hpp"
#include "optix/ProgramGroupBuilder.hpp"
#include "optix/SbtBuilder.hpp"
#include "optix/LaunchManager.hpp"
#include "optix/CudaOptixHelpers.hpp"

#include "radiation/DataBuilder.hpp"
#include "radiation/RayArrayBuilder.hpp"
#include "radiation/RayGenerator.hpp"
#include "radiation/ViewFactor.hpp"
#include "radiation/RadiativeFlux.hpp"
#include "radiation/Launcher.hpp"

#include "linear_algebra/SparseMatrix.hpp"
#include "linear_algebra/SparseLinearSystem.hpp"
#include "linear_algebra/LinearVector.hpp"
#include "linear_algebra/GaussSeidelSolver.hpp"
#include "linear_algebra/GaussSeidelSmoother.hpp"

using namespace std;

void Launcher::launchSimulation(){
    
    // Initialisation

    optixInit();
    cudaFree(0);
    OptixDeviceContext context = nullptr;
    OptixDeviceContextOptions options = {};
    optixDeviceContextCreate(0, &options, &context);
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    paths_.validateInputCase(caseName_, subCaseName_);
    paths_.createOutputCaseDir(caseName_, subCaseName_);

    cout << "Using input case: " << caseName_ << "/" << subCaseName_ << endl;
    cout << "Surface A file: " << paths_.surfaceAFile(caseName_, subCaseName_) << endl;
    cout << "Surface B file: " << paths_.surfaceBFile(caseName_, subCaseName_) << endl;
    cout << "Output directory: " << paths_.outputCaseDir(caseName_, subCaseName_) << endl;

    //vector<string> file = {fileA, fileB} ;
    //vector<string> propertyfile = {propertyFileA, propertyFileB} ;
    //DataBuilder dataBuilder_(file, propertyfile, psiNum_, thetaNum_) ; 
    //ViewFactor viewFactor_(dataBuilder_.triangleProperties().size()) ;

    // Batching 

    int numTotalTriangles = dataBuilder_.triangleProperties().size() ;
    int batchTriangleCount = batchTriangleCount_;

    // OUTSIDE LOOP

    // GAS Handle creation
    GasBuilder surfaceGasA ;
    GasBuilder surfaceGasB ;
    surfaceGasA.buildTriangleGas(context, stream, dataBuilder_.verticesA(), dataBuilder_.indicesA()) ;
    surfaceGasB.buildTriangleGas(context, stream, dataBuilder_.verticesB(), dataBuilder_.indicesB()) ;
    cudaStreamSynchronize(stream);

    cout << "GAS A build finished\n";
    cout << "GAS B build finished\n";
    cout << "surfaceGasA.handle = " << static_cast<unsigned long long>(surfaceGasA.handle()) << "\n";
    cout << "surfaceGasB.handle = " << static_cast<unsigned long long>(surfaceGasB.handle()) << "\n";

    // IAS handle creation
    IasBuilder ias ;
    ias.buildIas(context, stream, surfaceGasA, surfaceGasB) ;
    cout << "IAS build finished\n";
    cout << "ias.handle = " << static_cast<unsigned long long>(ias.handle()) << "\n";

    // Device program file initialisation
    cout << "Before opening PTX" << endl;
    ifstream ptxFile("devicePrograms.ptx");
    if (!ptxFile){
        cerr << "Could not open devicePrograms.ptx\n";
        return ;
    }
    cout << "Before reading PTX" << endl;
    stringstream ptxBuffer;
    ptxBuffer << ptxFile.rdbuf();
    string ptxCode = ptxBuffer.str();

    cout << "PTX size = " << ptxCode.size() << " bytes" << endl;
    cout << "Before module creation" << endl;

    // Pipeline compile option initialisation
    PipelineBuilder pipelineBuilder ;
    OptixPipelineCompileOptions pipelineCompileOptions = pipelineBuilder.pipelineCompileOptions() ;

    // Module Builder
    ModuleBuilder moduleBuilder ;
    moduleBuilder.createModule(context, pipelineCompileOptions, ptxCode) ; 

    cout << "After module creation" << endl;
    
    // Program groups
    ProgramGroupBuilder programGroupBuilder(context, moduleBuilder.module()) ;
    OptixProgramGroup programGroups[] = {programGroupBuilder.raygenPG(), programGroupBuilder.missPG(), programGroupBuilder.hitPG()};

    // Pipeline creation
    OptixResult resPipeline = pipelineBuilder.createOptixPipeline(context, programGroups) ;
    cout << "Pipeline log:\n" << pipelineBuilder.pipelineLog() << "\n";
    if (resPipeline != OPTIX_SUCCESS){
        cerr << "Pipeline creation failed\n";
        return ;
    }
    cout << "Pipeline created successfully\n";

    // SBT record evaluation
    SbtBuilder sbtBuilder(programGroupBuilder.raygenPG(), programGroupBuilder.missPG(), programGroupBuilder.hitPG()) ;
    cout << "SBT created successfully\n";

    OptixResult resStack = pipelineBuilder.setOptixPipelineStackSize() ;
    if (resStack != OPTIX_SUCCESS){
        cerr << "Pipeline stack size setup failed\n";
        return ;
    }
    cout << "Pipeline stack size set successfully\n";

    // OUTSIDE THE LOOP

    for(int batchStartTriangle = 0; batchStartTriangle < numTotalTriangles; batchStartTriangle += batchTriangleCount){

        int batchEndTriangle = min(static_cast<int>(numTotalTriangles), static_cast<int>(batchStartTriangle + batchTriangleCount)) ;
        
        dataBuilder_.buildRayDataForTriangleBatch(batchStartTriangle, batchEndTriangle) ;
        cout << "Number of rays in this batch: " << dataBuilder_.rayOrigins().size() << endl;
        cout << "rayDirections size: " << dataBuilder_.rayDirections().size() << endl;
        cout << "originGlobalTriangleIds size: " << dataBuilder_.originGlobalTriangleIds().size() << endl;
        if(dataBuilder_.rayOrigins().empty()){ throw runtime_error("No rays were generated for this batch"); }

        // Ray and triangle launch parameter evaluation
        LaunchManager launchManager(dataBuilder_, ias) ;
        cout << "Launch parameters prepared successfully\n";

        // Launch code 
        OptixResult resLaunch = launchManager.LaunchOptixSimulation(pipelineBuilder, sbtBuilder, stream) ;
        
        // View factor calculation and result display
        viewFactor_.accumulateBatch(launchManager, stream) ;
    }

    ofstream rout(paths_.outputFile(caseName_, subCaseName_, "progress.txt")) ;
    viewFactor_.finaliseViewFactors(rout) ;
    
    radiativeFlux_ = std::make_unique<RadiativeFlux>(viewFactor_, dataBuilder_);
    radiativeFlux_->initializeMatrices();
    radiativeFlux_->calculateHeatTransfer();

}