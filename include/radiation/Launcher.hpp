# pragma once 

#include <iostream>
#include <vector>
#include <array>
#include <algorithm>
#include <cmath>
#include <memory>

#include <cuda.h>
#include <cuda_runtime.h>

#include <optix.h>
#include <optix_stubs.h>
#include <optix_device.h>

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

#include "linear_algebra/SparseMatrix.hpp"
#include "linear_algebra/SparseLinearSystem.hpp"
#include "linear_algebra/LinearVector.hpp"
#include "linear_algebra/GaussSeidelSolver.hpp"
#include "linear_algebra/GaussSeidelSmoother.hpp"

using namespace std;

class Launcher{

    private:

        PathConfig& paths_ ;

        string caseName_ ;
        string subCaseName_ ;
        int psiNum_ ;
        int thetaNum_ ;
        int batchTriangleCount_ ;

        string fileA_ ;
        string fileB_ ;
        string propertyFileA_ ;
        string propertyFileB_ ;

        DataBuilder dataBuilder_ ;
        ViewFactor viewFactor_ ;
        std::unique_ptr<RadiativeFlux> radiativeFlux_;


    public:

        // Constructs the full DTM simulation driver for a selected case, subcase, angular resolution, and batch size
        Launcher(PathConfig& paths, string caseName, string subCaseName, int psiNum, int thetaNum, int batchTriangleCount)
                :paths_(paths),
                 caseName_(caseName),
                 subCaseName_(subCaseName),
                 psiNum_(psiNum),
                 thetaNum_(thetaNum),
                 batchTriangleCount_(batchTriangleCount),
                 fileA_(paths_.surfaceAFile(caseName_, subCaseName_)),
                 fileB_(paths_.surfaceBFile(caseName_, subCaseName_)),
                 propertyFileA_(paths_.surfaceAPropertiesFile(caseName_, subCaseName_)),
                 propertyFileB_(paths_.surfaceBPropertiesFile(caseName_, subCaseName_)),
                 dataBuilder_(DataBuilder({fileA_, fileB_}, {propertyFileA_, propertyFileB_}, psiNum_, thetaNum_)),
                 viewFactor_(ViewFactor(dataBuilder_.triangleProperties().size())),
                 radiativeFlux_(nullptr)
                 {}

        // Returns the completed radiative-flux object in read-only form
        const RadiativeFlux& radiativeFlux() const { return *radiativeFlux_; }

        // Returns the completed radiative-flux object for modification
        RadiativeFlux& radiativeFlux() { return *radiativeFlux_; }

        // Returns the input geometry and property data builder in read-only form
        const DataBuilder& dataBuilder() const{ return dataBuilder_ ; }

        // Returns the input geometry and property data builder for modification
        DataBuilder& dataBuilder(){ return dataBuilder_ ; }

        // Returns the computed view-factor object in read-only form
        const ViewFactor& viewFactor() const{ return viewFactor_ ; }

        // Returns the computed view-factor object for modification
        ViewFactor& viewFactor(){ return viewFactor_ ; }

        // Executes the complete DTM workflow, including data preparation, OptiX ray tracing, view-factor construction, and radiative-flux calculation
        void launchSimulation() ;

} ;