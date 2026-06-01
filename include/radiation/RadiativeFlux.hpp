# pragma once

#include <iostream>
#include <vector>
#include <array>
#include <fstream>
#include <string>
#include <sstream>
#include <map>
#include <unordered_map>

#include "radiation/ViewFactor.hpp"
#include "radiation/DataBuilder.hpp"

#include "linear_algebra/SparseLinearSystem.hpp"
#include "linear_algebra/GaussSeidelSolver.hpp"


class RadiativeFlux{

    private:

        ViewFactor& viewFactor_ ;
        DataBuilder& dataBuilder_ ;
        int n_ ;

        SparseLinearSystem Dg_ ;
        SparseMatrix Ft_ ;
        LinearVector Ie_ ;
        LinearVector A_ ;
        LinearVector T_ ;

        LinearVector J_ ; // Radiosity
        LinearVector G_ ; // Irradiation

        const double sigma = 5.670374419e-8;
        GaussSeidelSolver solver_ ;

        LinearVector q_ ;
        LinearVector Q_ ;

        double totalqA_ ;
        double totalqB_ ;

        double totalQA_ ;
        double totalQB_ ;

        double totalAreaA_ ;
        double totalAreaB_ ;

        double grossHeatTransferAtoB_ ;
        double grossHeatTransferBtoA_ ;
        double netHeatTransferAtoB_ ;

        double heatLossSurroundingA_ ;
        double heatLossSurroundingB_ ;
        double heatLossSurroundingFluxA_ ;
        double heatLossSurroundingFluxB_ ;

        double netPlateExchangeFluxA_ ;
        double netPlateExchangeFluxB_ ;

        double numTrianglesInA_ = 0 ;
        double numTrianglesInB_ = 0 ;

        double simulationRuntime_ = 0 ;

        LinearVector viewFactorRowSum_ ;
        double maxViewFactorRowSum_ ;
        double minViewFactorRowSum_ ;


    public:

        // Constructs the radiative-flux solver using the computed view factors and triangle property data.
        RadiativeFlux(ViewFactor& viewFactor, DataBuilder& dataBuilder):viewFactor_(viewFactor),
                                                                        dataBuilder_(dataBuilder),
                                                                        n_(dataBuilder.indicesA().size() + dataBuilder.indicesB().size()),
                                                                        Dg_(n_),
                                                                        Ft_(viewFactor.viewFactors()),
                                                                        Ie_(n_),
                                                                        A_(n_),
                                                                        T_(n_),
                                                                        J_(n_),
                                                                        G_(n_),
                                                                        solver_(Dg_),
                                                                        q_(n_),
                                                                        Q_(n_),
                                                                        viewFactorRowSum_(n_)
        {
            
        }

        // Returns the radiosity linear system in read-only form
        const SparseLinearSystem& Dg() const{ return this->Dg_ ;}

        // Returns the radiosity linear system for modification
        SparseLinearSystem& Dg(){return this->Dg_ ;}

        // Returns the transposed or transformed view-factor matrix in read-only form.
        const SparseMatrix& Ft() const{ return this->Ft_ ;}

        // Returns the transposed or transformed view-factor matrix for modification.
        SparseMatrix& Ft() { return this->Ft_ ;}

        // Returns the blackbody emissive-power vector in read-only form
        const LinearVector& Ie() const{ return this->Ie_ ;}

        // Returns the blackbody emissive-power vector for modification
        LinearVector& Ie() { return this->Ie_ ;}

        // Returns the triangle area vector in read-only form
        const LinearVector& A() const{ return this->A_ ;}

        // Returns the triangle area vector for modification
        LinearVector& A(){ return this->A_ ;}

        // Returns the triangle temperature vector in read-only form
        const LinearVector& T() const{ return this->T_ ;}

        // Returns the triangle temperature vector for modification
        LinearVector& T(){return this->T_ ;}

        // Returns the radiosity vector in read-only form
        const LinearVector& J() const{ return this->J_ ;}

        // Returns the radiosity vector for modification
        LinearVector& J(){return this->J_;}

        // Returns the irradiation vector in read-only form
        const LinearVector& G() const{ return this->G_ ;}

        // Returns the irradiation vector for modification
        LinearVector& G(){return this->G_;}

        // Returns the surface heat-flux vector in read-only form
        const LinearVector& q() const{ return this->q_ ;}

        // Returns the surface heat-flux vector for modification
        LinearVector& q() {return this->q_ ;}

        // Returns the total heat-transfer-rate vector in read-only form
        const LinearVector& Q() const{ return this->Q_ ;}

        // Returns the total heat-transfer-rate vector for modification
        LinearVector& Q(){ return this->Q_ ;}

        // Returns the area-averaged heat flux over surface A
        double totalqA(){return this->totalqA_ ;}

        // Returns the area-averaged heat flux over surface B
        double totalqB(){return this->totalqB_ ;}

        // Returns the total heat-transfer rate over surface A
        double totalQA(){return this->totalQA_ ;}

        // Returns the total heat-transfer rate over surface B
        double totalQB(){return this->totalQB_ ;}

        // Returns the total discretised area of surface A
        double totalAreaA(){return this->totalAreaA_ ;}

        // Returns the total discretised area of surface B
        double totalAreaB(){return this->totalAreaB_ ;}

        // Returns the gross radiative heat transfer from surface A to surface B
        double grossHeatTransferAtoB(){return this->grossHeatTransferAtoB_ ;}

        // Returns the gross radiative heat transfer from surface B to surface A
        double grossHeatTransferBtoA(){return this->grossHeatTransferBtoA_ ;}

        // Returns the net radiative heat transfer from surface A to surface B
        double netHeatTransferAtoB(){return this->netHeatTransferAtoB_ ;}

        // Returns the radiative heat loss from surface A to the surroundings
        double heatLossSurroundingA(){return this->heatLossSurroundingA_ ;}

        // Returns the radiative heat loss from surface B to the surroundings
        double heatLossSurroundingB(){return this->heatLossSurroundingB_ ;}

        // Returns the area-averaged surrounding-loss heat flux from surface A
        double heatLossSurroundingFluxA(){return this->heatLossSurroundingFluxA_ ;}

        // Returns the area-averaged surrounding-loss heat flux from surface B
        double heatLossSurroundingFluxB(){return this->heatLossSurroundingFluxB_ ;}

        // Returns the net plate-to-plate exchange flux evaluated from surface A
        double netPlateExchangeFluxA(){return this->netPlateExchangeFluxA_ ;}

        // Returns the net plate-to-plate exchange flux evaluated from surface B
        double netPlateExchangeFluxB(){return this->netPlateExchangeFluxB_ ;}

        // Returns the total number of triangles included in the radiative-flux calculation
        int n(){ return this->n_ ;}
         
        // Returns the view-factor row-sum vector in read-only form
        const LinearVector& viewFactorRowSum() const {return viewFactorRowSum_ ;}

        // Returns the view-factor row-sum vector for modification
        LinearVector& viewFactorRowSum(){return viewFactorRowSum_ ;}

        // Returns the number of triangles belonging to surface A
        int numTrianglesInA(){ return this->numTrianglesInA_ ;}

        // Returns the number of triangles belonging to surface B
        int numTrianglesInB(){ return this->numTrianglesInB_ ;}

        // Returns the maximum row sum of the view-factor matrix
        double maxViewFactorRowSum(){ return maxViewFactorRowSum_ ;}

        // Returns the minimum row sum of the view-factor matrix.
        double minViewFactorRowSum(){ return minViewFactorRowSum_ ;}

        // Stores the number of triangles belonging to surface A
        void setNumTrianglesInA(int numTrianglesInA){ this->numTrianglesInA_ = numTrianglesInA ;}

        // Stores the number of triangles belonging to surface B
        void setNumTrianglesInB(int numTrianglesInB){ this->numTrianglesInB_ = numTrianglesInB ;}

        // Stores the maximum row sum of the view-factor matrix
        void setMaxViewFactorRowSum(double maxViewFactorRowSum){this->maxViewFactorRowSum_ = maxViewFactorRowSum ;}

        // Stores the minimum row sum of the view-factor matrix
        void setMinViewFactorRowSum(double minViewFactorRowSum){this->minViewFactorRowSum_ = minViewFactorRowSum ;}

        // Returns the measured OptiX simulation runtime
        double simulationRuntime(){ return this->simulationRuntime_ ;}

        // Stores the measured OptiX simulation runtime
        void setSimulationRuntime(double simulationRuntime){ this->simulationRuntime_ = simulationRuntime ;}

        // Builds the view-factor matrix used for irradiation and radiosity calculations
        void buildFtMatrix() ;

        // Builds the blackbody emissive-power vector from triangle temperatures
        void buildIeVector() ;

        // Builds the triangle area vector from the surface property data
        void buildAVector() ;

        // Replaces each triangle area with its reciprocal for area-weighted matrix operations
        void reciprocateAVector() ;

        // Builds the triangle temperature vector from the surface property data
        void buildTVector() ;

        // Forms the area-weighted precursor matrix A^{-1} F^T A used to compute irradiation from radiosity
        void calculatePrecursorMatrix() ;   // A^-1*F^T*A, where raw F(i,j)=F_i_to_j

        // Checks whether all surface triangles behave as blackbody emitters
        bool blackbody() ;

        // Computes radiosity directly from blackbody emissive power for blackbody surfaces
        void calculateJBlackBody() ;

        // Solves the diffuse-gray radiosity system for non-blackbody surfaces
        void calculateJDiffuseGray() ;

        // Computes irradiation from the radiosity field and view-factor coupling
        void calculateG() ;
        
        // Computes heat transfer for the special case where all surfaces are blackbodies
        void calculateBlackBodyHeatTransfer() ;

        // Computes heat transfer for diffuse-gray surfaces using radiosity and irradiation
        void calculateDiffuseGrayHeatTransfer() ;

        // Computes local heat fluxes and total heat-transfer rates from radiosity and irradiation
        void calculateHeatFluxAndMagnitudeTransfer() ;

        // Computes gross exchange, net exchange, and surrounding-loss diagnostics between the two surfaces
        void calculateExchangeDiagnostics() ;

        // Runs the full radiative heat-transfer calculation using the appropriate blackbody or diffuse-gray formulation
        void calculateHeatTransfer() ;

        // Initialises all matrices and vectors required for the radiative-flux calculation
        void initializeMatrices() ;

        // Runs a temporary diagnostic test routine for checking radiative-flux calculations
        void test() ;
} ;
