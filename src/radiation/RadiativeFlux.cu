# include <iostream>
# include <vector>
# include <array>
# include <fstream>
# include <string>
# include <sstream>
# include <map>
# include <unordered_map>

# include "radiation/ViewFactor.hpp"
# include "radiation/DataBuilder.hpp"
# include "radiation/RadiativeFlux.hpp"
# include "linear_algebra/SparseLinearSystem.hpp"

void RadiativeFlux::buildFtMatrix(){

    for(const coo& e : this->viewFactor_.viewFactors().entries()){

        int i = e.row;
        int j = e.col;
        double Fij = e.val;

        this->Ft().put(j, i, Fij);
        cout << "Ft_" << i << "_" << j << " :" << Fij << endl ;
    }
}

void RadiativeFlux::buildIeVector(){
    for(int i = 0; i < n_; ++i){
        this->Ie()(i) = 1 - this->dataBuilder_.triangleProperties()[i].emissivity ;
    }
}

void RadiativeFlux::buildAVector(){
    for(int i = 0; i < n_; ++i){
        this->A()(i) = this->dataBuilder_.triangleProperties()[i].area ;
    }
}

void RadiativeFlux::reciprocateAVector(){
    for(int i = 0; i < n_; ++i){
        this->A()(i) = 1/this->A()(i) ;
    }
}

void RadiativeFlux::buildTVector(){
    for(int i = 0; i < n_; ++i){
        this->T()(i) = this->dataBuilder_.triangleProperties()[i].temperature ;
    }
}

bool RadiativeFlux::blackbody(){
    for(int i = 0; i < n_; ++i){
        if(this->Ie()(i) !=0 ){
            return false ;
        }
    }
    return true ;
}

void RadiativeFlux::calculatePrecursorMatrix(){
    // viewFactor.viewFactors() stores the raw Monte-Carlo view factor as
    // row = emitter i, col = receiver j, value = F_{i->j}.
    // Irradiation needs G_i = sum_j (A_j/A_i) F_{j->i} J_j.
    // Therefore Ft_ must become A^{-1} F^T A before calculateG().
    LinearVector invA(n_);
    for(int i = 0; i < n_; ++i){
        invA(i) = 1.0 / this->A()(i);
    }

    this->Ft().transpose();
    this->Ft().rightMultiplyMatrix(this->A());
    this->Ft().leftMultiplyMatrix(invA);
}

void RadiativeFlux::calculateJBlackBody(){
    for(int i = 0; i < n_; ++i){
        this->J()(i) = this->dataBuilder_.triangleProperties()[i].emissivity*this->sigma*pow(this->T()(i),4) ;
    }
}

void RadiativeFlux::calculateJDiffuseGray(){

    this->Dg().A() = this->Ft() ;
    this->Dg().A().leftMultiplyMatrix(this->Ie()) ;
    this->Dg().A().substractFromIdentityMatrix() ;

    for(int i = 0; i < n_; ++i){
        this->Dg().b()(i) = this->dataBuilder_.triangleProperties()[i].emissivity*this->sigma*pow(this->T()(i),4) ;
    }

    this->solver_.solve(0.000001, true) ;

    for(int i = 0; i < n_; ++i){
        this->J()(i) = this->Dg().X()(i) ;
    }

}

void RadiativeFlux::calculateG(){
    this->G() = this->Ft()*this->J() ;
}
        
void RadiativeFlux::calculateBlackBodyHeatTransfer(){
    this->calculateJBlackBody() ;
    cout << "Calculated J black body" << endl ;
    this->calculateG() ;
    cout << "Calculated G" << endl ;
    this->calculateHeatFluxAndMagnitudeTransfer() ;
    this->calculateExchangeDiagnostics() ;
    cout << "Calculated heat flux and transfer" << endl ;
}

void RadiativeFlux::calculateDiffuseGrayHeatTransfer(){
    this->calculateJDiffuseGray() ;
    this->calculateG() ;
    this->calculateHeatFluxAndMagnitudeTransfer() ;
    this->calculateExchangeDiagnostics() ;
}

void RadiativeFlux::calculateHeatFluxAndMagnitudeTransfer(){

    this->q() = this->J()-this->G() ;
    this->Q() = this->A()*this->q() ;
    
    this->totalQA_ = 0 ;
    this->totalQB_ = 0 ;

    this->totalqA_ = 0 ;
    this->totalqB_ = 0 ;

    this->totalAreaA_ = 0 ;
    this->totalAreaB_ = 0 ;

    for(int i=0; i<dataBuilder_.indicesA().size(); ++i){
        this->totalQA_ += this->Q()(i) ;
        this->totalAreaA_ += this->A()(i) ;
    }

    for(int i=dataBuilder_.indicesA().size(); i<this->n_; ++i){
        this->totalQB_ += this->Q()(i) ;
        this->totalAreaB_ += this->A()(i) ;
    }

    this->totalqA_ = this->totalQA_ / this->totalAreaA_ ;
    this->totalqB_ = this->totalQB_ / this->totalAreaB_ ;
}

void RadiativeFlux::calculateExchangeDiagnostics(){

    const int numA = static_cast<int>(this->dataBuilder_.indicesA().size());

    this->grossHeatTransferAtoB_ = 0.0;
    this->grossHeatTransferBtoA_ = 0.0;
    this->heatLossSurroundingA_ = 0.0;
    this->heatLossSurroundingB_ = 0.0;

    // Use the raw view-factor matrix here, not Ft_, because Ft_ has been
    // converted to the irradiation matrix A^{-1}F^T A.
    for(const coo& e : this->viewFactor_.viewFactors().entries()){

        const int i = e.row;      // emitter
        const int j = e.col;      // receiver
        const double Fij = e.val; // F_{i->j}

        const double Qij = this->A()(i) * Fij * this->J()(i);

        if(i < numA && j >= numA){
            this->grossHeatTransferAtoB_ += Qij;
        }
        else if(i >= numA && j < numA){
            this->grossHeatTransferBtoA_ += Qij;
        }
    }

    for(int i = 0; i < this->n_; ++i){
        double FiSurr = 1.0 - viewFactor_.viewFactorRowSum()(i);

        // Small negative values can occur from Monte-Carlo/noise or round-off.
        if(FiSurr < 0.0 && FiSurr > -1.0e-12){
            FiSurr = 0.0;
        }

        const double QiSurr = this->A()(i) * FiSurr * this->J()(i);

        if(i < numA){
            this->heatLossSurroundingA_ += QiSurr;
        }
        else{
            this->heatLossSurroundingB_ += QiSurr;
        }
    }

    this->netHeatTransferAtoB_ = this->grossHeatTransferAtoB_ - this->grossHeatTransferBtoA_;

    this->heatLossSurroundingFluxA_ = this->heatLossSurroundingA_ / this->totalAreaA_;
    this->heatLossSurroundingFluxB_ = this->heatLossSurroundingB_ / this->totalAreaB_;

    this->netPlateExchangeFluxA_ = this->netHeatTransferAtoB_ / this->totalAreaA_;
    this->netPlateExchangeFluxB_ = -this->netHeatTransferAtoB_ / this->totalAreaB_;
}

void RadiativeFlux::calculateHeatTransfer(){

    if(blackbody()){
        this->calculateBlackBodyHeatTransfer() ;
    }
    else{
        this->calculateDiffuseGrayHeatTransfer() ;
    }

}

void RadiativeFlux::initializeMatrices(){

    this->buildIeVector() ;
    cout << " Built the IE vector " << endl ;

    this->buildAVector() ;
    cout << " Built the A Vector " << endl ;

    this->buildTVector() ;
    cout << " Built the T vector " << endl ;
    
    this->calculatePrecursorMatrix() ;
    cout << " Built the P matrix " << endl ;

}

void RadiativeFlux::test(){

    ofstream out("/home/raid/as3736/DTMTest/Version5/Output/100.txt") ;

    out << "Ft: " ;
    for(int i=0; i<n_; ++i){
        for(int j=0; j<n_; ++j){
            out << this->Ft()(i,j) << " ";
        }
        out << endl; 
    }
    out << endl; 

    out << endl << "Ie: " ;
    for(int i=0; i<n_; ++i){
        out << this->Ie()(i) << " " ;
    }
    out << endl;

    out << endl << "A: " ;
    for(int i=0; i<n_; ++i){
        out << this->A()(i) << " " ;
    }   
    out << endl;

    out << endl << "T: ";
    for(int i=0; i<n_; ++i){
        out << this->T()(i) << " " ;
    }  
    out << endl;

    this->calculatePrecursorMatrix() ;

    out << "Precursor Ft: " ;
    for(int i=0; i<n_; ++i){
        for(int j=0; j<n_; ++j){
            out << this->Ft()(i,j) << " " ;
        }
        out << endl; 
    }
    out << endl;

    this->calculateJDiffuseGray() ;
    out << endl << "J: "  ;
    for(int i=0; i<n_; ++i){
        out << this->J()(i) << " " ;
    }  
    out << endl;

    this->calculateG() ;
    out << endl << "G: "  ;
    for(int i=0; i<n_; ++i){
        out << this->G()(i) << " " ;
    }  
    out << endl;

}

