#include <iostream>
#include <vector>
#include <array>
#include <algorithm>
#include <cmath>
#include <chrono>

#include <cuda.h>
#include <cuda_runtime.h>

#include <optix.h>
#include <optix_stubs.h>
#include <optix_device.h>

#include <fstream>
#include <sstream>
#include <string>

#include "core/Triangle.hpp"
#include "core/RayGeometry.hpp"
#include "core/RayParams.hpp"
#include "core/PathConfig.hpp"
#include "core/AnalyticalEvaluator.hpp"

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

void diagnostics(ofstream& out, RadiativeFlux& radiativeFlux){

    out << endl << endl << "Surface A(J): " << endl ;
    for(int i=0; i < radiativeFlux.numTrianglesInA(); ++i){ out << radiativeFlux.J()(i) << " " ; }
    out << endl ;

    out << "Surface A(G): " << endl ;
    for(int i=0; i < radiativeFlux.numTrianglesInA(); ++i){ out << radiativeFlux.G()(i) << " " ; }
    out << endl ;

    out << "Surface B(J): " << endl ;
    for(int i=radiativeFlux.numTrianglesInA(); i < radiativeFlux.n(); ++i){ out << radiativeFlux.J()(i) << " " ; }
    out << endl ;

    out << "Surface B(G): " << endl ;
    for(int i=radiativeFlux.numTrianglesInA(); i < radiativeFlux.n(); ++i){ out << radiativeFlux.G()(i) << " " ; }
    out << endl ;

    out << endl;
    out << "SIGN CONVENTION:" << endl;
    out << "  q = J - G is net radiative heat flux LEAVING the surface." << endl;
    out << "  Positive total Q/q means net loss by the plate; negative means net gain by the plate." << endl;
    out << "  Heat loss to surroundings assumes open black surroundings at 0 K." << endl;

    out << endl << "1. TOTAL NET HEAT FLUX ON EACH PLATE" << endl;
    out << "Plate A total net heat rate, QA_total = " << radiativeFlux.totalQA() << " W" << endl ;
    out << "Plate B total net heat rate, QB_total = " << radiativeFlux.totalQB() << " W" << endl ;
    out << "Plate A total net heat flux, qA_total = " << radiativeFlux.totalqA() << " W/m2" << endl ;
    out << "Plate B total net heat flux, qB_total = " << radiativeFlux.totalqB() << " W/m2" << endl ;

    out << endl << "2. HEAT LOSS TO SURROUNDINGS" << endl;
    out << "Plate A heat loss to surroundings, QA_surr = " << radiativeFlux.heatLossSurroundingA() << " W" << endl ;
    out << "Plate B heat loss to surroundings, QB_surr = " << radiativeFlux.heatLossSurroundingB() << " W" << endl ;
    out << "Plate A heat-loss flux to surroundings, qA_surr = " << radiativeFlux.heatLossSurroundingFluxA() << " W/m2" << endl ;
    out << "Plate B heat-loss flux to surroundings, qB_surr = " << radiativeFlux.heatLossSurroundingFluxB() << " W/m2" << endl ;

    out << endl << "3. HEAT TRANSFER BETWEEN BOTH PLATES" << endl;
    out << "Gross heat transfer A to B, QA_to_B = " << radiativeFlux.grossHeatTransferAtoB() << " W" << endl ;
    out << "Gross heat transfer B to A, QB_to_A = " << radiativeFlux.grossHeatTransferBtoA() << " W" << endl ;
    out << "Net heat transfer A to B, Qnet_A_to_B = " << radiativeFlux.netHeatTransferAtoB() << " W" << endl ;
    out << "Plate-to-plate net exchange flux on A = " << radiativeFlux.netPlateExchangeFluxA() << " W/m2" << endl ;
    out << "Plate-to-plate net exchange flux on B = " << radiativeFlux.netPlateExchangeFluxB() << " W/m2" << endl ;

    out << endl << "CHECKS" << endl;
    out << "QA_total should equal Qnet_A_to_B + QA_surr = " << radiativeFlux.netHeatTransferAtoB() + radiativeFlux.heatLossSurroundingA() << " W" << endl;
    out << "QB_total should equal -Qnet_A_to_B + QB_surr = " << -radiativeFlux.netHeatTransferAtoB() + radiativeFlux.heatLossSurroundingB() << " W" << endl << endl;

    out << "Simulation Runtime: " << radiativeFlux.simulationRuntime() << " seconds" << endl ;

    out << "Maximum View Factor Row Sum: " << radiativeFlux.maxViewFactorRowSum() << endl ;
    out << "Minimum View Factor Row Sum: " << radiativeFlux.minViewFactorRowSum() << endl ;
}

RadiativeFlux run(PathConfig& paths_, string caseName, string subCaseName, int batchTriangleCount, int thetaNum, int psiNum){
    
    //PathConfig paths_ = paths;
    string caseName_ = caseName ;
    string subCaseName_ = subCaseName ;
    //int batchTriangleCount_ = batchTriangleCount ;
    int psiNum_ = psiNum ;
    int thetaNum_ = thetaNum ;

    paths_.validateInputCase(caseName_, subCaseName_);
    paths_.createOutputCaseDir(caseName_, subCaseName_);

    optixInit();
    cudaFree(0);
    OptixDeviceContext context = nullptr;
    OptixDeviceContextOptions options = {};
    optixDeviceContextCreate(0, &options, &context);
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    //paths_.validateInputCase(caseName_, subCaseName_);
    //paths_.createOutputCaseDir(caseName_, subCaseName_);

    cout << "Using input case: " << caseName_ << "/" << subCaseName_ << endl;
    cout << "Surface A file: " << paths_.surfaceAFile(caseName_, subCaseName_) << endl;
    cout << "Surface B file: " << paths_.surfaceBFile(caseName_, subCaseName_) << endl;
    cout << "Output directory: " << paths_.outputCaseDir(caseName_, subCaseName_) << endl;

    string fileA_ = paths_.surfaceAFile(caseName_, subCaseName_) ;
    string fileB_ = paths_.surfaceBFile(caseName_, subCaseName_) ;
    string propertyFileA_ = paths_.surfaceAPropertiesFile(caseName_, subCaseName_) ;
    string propertyFileB_ = paths_.surfaceBPropertiesFile(caseName_, subCaseName_) ;

    vector<string> file = {fileA_, fileB_} ;
    vector<string> propertyfile = {propertyFileA_, propertyFileB_} ;
    DataBuilder dataBuilder_(file, propertyfile, thetaNum_, psiNum_) ; 
    ViewFactor viewFactor_(dataBuilder_.triangleProperties().size()) ;

    // Batching 

    int numTotalTriangles = dataBuilder_.triangleProperties().size() ;
    //int batchTriangleCount = batchTriangleCount_;

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
    }
    cout << "Pipeline created successfully\n";

    // SBT record evaluation
    SbtBuilder sbtBuilder(programGroupBuilder.raygenPG(), programGroupBuilder.missPG(), programGroupBuilder.hitPG()) ;
    cout << "SBT created successfully\n";

    OptixResult resStack = pipelineBuilder.setOptixPipelineStackSize() ;
    if (resStack != OPTIX_SUCCESS){
        cerr << "Pipeline stack size setup failed\n";
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
    
    RadiativeFlux radiativeFlux_(viewFactor_, dataBuilder_);
    radiativeFlux_.initializeMatrices();
    radiativeFlux_.calculateHeatTransfer();

    radiativeFlux_.viewFactorRowSum() = viewFactor_.viewFactorRowSum() ;

    radiativeFlux_.setNumTrianglesInA(dataBuilder_.indicesA().size()) ;
    radiativeFlux_.setNumTrianglesInB(dataBuilder_.indicesB().size()) ;
    radiativeFlux_.setMaxViewFactorRowSum(viewFactor_.maxViewFactorRowSum()) ;
    radiativeFlux_.setMinViewFactorRowSum(viewFactor_.minViewFactorRowSum()) ;

    cudaStreamDestroy(stream);
    optixDeviceContextDestroy(context);

    return radiativeFlux_ ;

}

// Heat flux test 
void heatFluxVariationTest(){

    string caseName = "heat_flux_test" ;
    vector<string> subCaseNames = {"300x300_gap0p01_0", "300x300_gap0p01_100", "300x300_gap0p01_200", "300x300_gap0p01_300", "300x300_gap0p01_400", "300x300_gap0p01_500", "300x300_gap0p01_600", "300x300_gap0p01_700", "300x300_gap0p01_800", "300x300_gap0p01_900", "300x300_gap0p01_1000"} ;

    vector<double> eastWallTemperature = {0,100,200,300,400,500,600,700,800,900,1000} ;
    double westWallTemperature = 1000 ;

    PathConfig paths ;
    paths.createOutputCaseDir(caseName);
    int psinum = 100 ;
    int thetanum = 500 ;
    int batchtrianglecount = 200 ;

    AnalyticalEvaluator analyticalValue(1, 0.01, 1, 1, 1.0, 1.0) ;

    ofstream heatFluxOut(paths.outputFile(caseName, "heat_flux_diagnostics.csv")) ;
    heatFluxOut << "west_wall_temperature,"
            << "east_wall_temperature,"
            << "heat_flux_A,"
            << "heat_flux_analytical_A,"
            << "heat_flux_B,"
            << "heat_flux_analytical_B,"
            << "heat_rate_A,"
            << "heat_rate_A_analytical,"
            << "heat_rate_B,"
            << "heat_rate_B_analytical,"
            << "heat_loss_surrounding_A,"
            << "heat_loss_surrounding_A_analytical,"
            << "heat_loss_surrounding_B,"
            << "heat_loss_surrounding_B_analytical,"
            << "heat_loss_flux_surrounding_A,"
            << "heat_loss_flux_surrounding_A_analytical,"
            << "heat_loss_flux_surrounding_B,"
            << "heat_loss_flux_surrounding_B_analytical,"
            << "gross_heat_transfer_AB,"
            << "gross_heat_transfer_AB_analytical,"
            << "gross_heat_transfer_BA,"
            << "gross_heat_transfer_BA_analytical,"
            << "net_heat_transfer,"
            << "net_heat_transfer_analytical,"
            << "net_plate_exchange_flux_A,"
            << "net_plate_exchange_flux_B,"
            << "min_view_factor_row_sum,"
            << "max_view_factor_row_sum" << endl;

    int i = 0 ;
    for(string subCaseName: subCaseNames){

        analyticalValue.calculateHeatFluxAndTransfer(westWallTemperature, eastWallTemperature[i]) ;

        auto start = std::chrono::high_resolution_clock::now();
        RadiativeFlux radiativeFlux = run(paths, caseName, subCaseName, batchtrianglecount, thetanum, psinum) ;
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;

        radiativeFlux.setSimulationRuntime(elapsed.count()) ;
        ofstream out(paths.outputFile(caseName, subCaseName, "overall_subcase_diagnostic.txt")) ;
        diagnostics(out, radiativeFlux) ;

        heatFluxOut << westWallTemperature << "," 
                    << eastWallTemperature[i] << "," 
                    << radiativeFlux.totalqA() << "," 
                    << analyticalValue.q1Total() << ","
                    << radiativeFlux.totalqB() << "," 
                    << analyticalValue.q2Total() << ","
                    << radiativeFlux.totalQA() << "," 
                    << analyticalValue.Q1Total() << ","
                    << radiativeFlux.totalQB() << "," 
                    << analyticalValue.Q2Total() << ","
                    << radiativeFlux.heatLossSurroundingA() << ","
                    << analyticalValue.Q1Surrounding() << ","
                    << radiativeFlux.heatLossSurroundingB() << ","
                    << analyticalValue.Q2Surrounding() << ","
                    << radiativeFlux.heatLossSurroundingFluxA() << ","
                    << analyticalValue.q1Surrounding() << ","
                    << radiativeFlux.heatLossSurroundingFluxB() << ","
                    << analyticalValue.q2Surrounding() << ","
                    << radiativeFlux.grossHeatTransferAtoB() << "," 
                    << analyticalValue.Q12Gross() << ","
                    << radiativeFlux.grossHeatTransferBtoA() << "," 
                    << analyticalValue.Q21Gross() << ","
                    << radiativeFlux.netHeatTransferAtoB() << "," 
                    << analyticalValue.Q12Net() << ","
                    << radiativeFlux.netPlateExchangeFluxA() << "," 
                    << radiativeFlux.netPlateExchangeFluxB() << "," 
                    << radiativeFlux.minViewFactorRowSum() << "," 
                    << radiativeFlux.maxViewFactorRowSum() << endl ;
        ++i ;
    }

}

// Convergence test and triangle runtime test
void convergenceAndTriangleRuntimeTest(){

    string caseName = "convergence_test_triangle_runtime_scaling" ;
    vector<string> subCaseNames = {"20x20_gap0p01", "50x50_gap0p01", "75x100_gap0p01", "100x100_gap0p01", "200x200_gap0p01", "300x300_gap0p01", "400x400_gap0p01"} ;
    vector<string> trianglesPerSurface = {"20","5000","15000","20000","80000","180000","320000"} ;

    double eastWallTemperature = 1000;
    double westWallTemperature = 1000 ;

    PathConfig paths ;
    paths.createOutputCaseDir(caseName);
    int psinum = 100 ;
    int thetanum = 500 ;
    int batchtrianglecount = 200 ;

    ofstream heatFluxOut(paths.outputFile(caseName, "heat_flux_diagnostics.csv")) ;
    heatFluxOut << "triangles_per_surface,runtime,west_wall_temperature,east_wall_temperature,heat_flux_A,heat_flux_B,heat_rate_A,heat_rate_B,heat_loss_surrounding_A,heat_loss_surrounding_B,heat_loss_flux_surrounding_A,heat_loss_flux_surrounding_B,gross_heat_transfer_AB,gross_heat_transfer_BA,net_heat_transfer,net_heat_exchange_A,net_heat_exchange_B,min_view_factor_row_sum,max_view_factor_row_sum" << endl ;

    int i = 0 ;
    for(string subCaseName: subCaseNames){

        auto start = std::chrono::high_resolution_clock::now();
        RadiativeFlux radiativeFlux = run(paths, caseName, subCaseName, batchtrianglecount, thetanum, psinum) ;
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;

        radiativeFlux.setSimulationRuntime(elapsed.count()) ;
        ofstream out(paths.outputFile(caseName, subCaseName, "overall_subcase_diagnostic.txt")) ;
        diagnostics(out, radiativeFlux) ;

        heatFluxOut << trianglesPerSurface[i] << "," << elapsed.count() << "," << westWallTemperature << "," << eastWallTemperature << "," << radiativeFlux.totalqA() << "," << radiativeFlux.totalqB() << "," << radiativeFlux.totalQA() << "," << radiativeFlux.totalQB() << "," << radiativeFlux.heatLossSurroundingA() << "," << radiativeFlux.heatLossSurroundingB() << "," << radiativeFlux.heatLossSurroundingFluxA() << "," << radiativeFlux.heatLossSurroundingFluxB() << "," << radiativeFlux.grossHeatTransferAtoB() << "," << radiativeFlux.grossHeatTransferBtoA() << "," << radiativeFlux.netHeatTransferAtoB() << "," << radiativeFlux.netPlateExchangeFluxA() << "," << radiativeFlux.netPlateExchangeFluxB() << "," << radiativeFlux.minViewFactorRowSum() << "," << radiativeFlux.maxViewFactorRowSum() << endl ;

        ++i ;
        
    }


}

// Ray runtime test
void rayRuntimeTest(){

    string caseName = "rays_runtime_scaling" ;
    string subCaseName = "300x300_gap0p01_500" ;

    double eastWallTemperature = 500 ;
    double westWallTemperature = 1000 ;

    PathConfig paths ;
    paths.createOutputCaseDir(caseName);
    int psinum = 100 ;
    vector<int> thetanums = {50,100,150,200,250,300,350,400,450,500} ;
    int batchtrianglecount = 200 ;

    ofstream heatFluxOut(paths.outputFile(caseName, "heat_flux_diagnostics.csv")) ;
    heatFluxOut << "ray_count,west_wall_temperature,east_wall_temperature,runtime,heat_flux_A,heat_flux_B,heat_rate_A,heat_rate_B,heat_loss_surrounding_A,heat_loss_surrounding_B,heat_loss_flux_surrounding_A,heat_loss_flux_surrounding_B,gross_heat_transfer_AB,gross_heat_transfer_BA,net_heat_transfer,net_heat_exchange_A,net_heat_exchange_B,min_view_factor_row_sum,max_view_factor_row_sum" << endl ;

    int i = 0 ;
    for(int thetanum: thetanums){

        auto start = std::chrono::high_resolution_clock::now();
        RadiativeFlux radiativeFlux = run(paths, caseName, subCaseName, batchtrianglecount, thetanum, psinum) ;
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;

        radiativeFlux.setSimulationRuntime(elapsed.count()) ;
        ofstream out(paths.outputFile(caseName, to_string(thetanum), "overall_subcase_diagnostic.txt")) ;
        diagnostics(out, radiativeFlux) ;

        heatFluxOut << to_string(thetanum*psinum) << "," << westWallTemperature << "," << eastWallTemperature << "," << elapsed.count() << "," << radiativeFlux.totalqA() << "," << radiativeFlux.totalqB() << "," << radiativeFlux.totalQA() << "," << radiativeFlux.totalQB() << "," << radiativeFlux.heatLossSurroundingA() << "," << radiativeFlux.heatLossSurroundingB() << "," << radiativeFlux.heatLossSurroundingFluxA() << "," << radiativeFlux.heatLossSurroundingFluxB() << "," << radiativeFlux.grossHeatTransferAtoB() << "," << radiativeFlux.grossHeatTransferBtoA() << "," << radiativeFlux.netHeatTransferAtoB() << "," << radiativeFlux.netPlateExchangeFluxA() << "," << radiativeFlux.netPlateExchangeFluxB() << "," << radiativeFlux.minViewFactorRowSum() << "," << radiativeFlux.maxViewFactorRowSum() << endl ;

        ++i ;
        
    }
}

int main(){

    heatFluxVariationTest() ;
    //convergenceAndTriangleRuntimeTest() ;
    //rayRuntimeTest() ;

}





