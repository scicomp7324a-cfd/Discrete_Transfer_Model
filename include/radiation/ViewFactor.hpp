# pragma once 

#include <iostream>
#include <vector>
#include <array>
#include <fstream>
#include <string>
#include <sstream>
#include <map>
#include <unordered_map>
#include <thrust/device_vector.h>

#include "optix/LaunchManager.hpp"
#include "linear_algebra/SparseMatrix.hpp"

class ViewFactor{

    private:

        int numTotalTriangles_ ;
        SparseMatrix viewFactors_;
        vector<TriangleTraceHistory> h_triangleHistoryOut_ ;

        vector<unsigned int> emits_;
        unordered_map<uint64_t, unsigned int> hitCounts_;        
        thrust::device_vector<unsigned int> d_emits_;

        LinearVector viewFactorRowSum_ ;
        double maxViewFactorRowSum_ ;
        double minViewFactorRowSum_ ;

    public:

        // Constructs the view-factor storage and tracing-history buffers for the total number of surface triangles
        ViewFactor(int numTotalTriangles):numTotalTriangles_(numTotalTriangles),
                                                 viewFactors_(numTotalTriangles),
                                                 h_triangleHistoryOut_(numTotalTriangles),
                                                 emits_(numTotalTriangles, 0),
                                                 d_emits_(numTotalTriangles, 0),
                                                 viewFactorRowSum_(numTotalTriangles)
        {
            hitCounts_.reserve(100000000);
            hitCounts_.max_load_factor(0.7);
        }

        // Returns the sparse view-factor matrix in read-only form
        const SparseMatrix& viewFactors() const { return viewFactors_; }

        // Returns the sparse view-factor matrix for modification
        SparseMatrix& viewFactors() { return viewFactors_; }

        // Returns the per-triangle row-sum vector of the computed view-factor matrix in read-only form
        const LinearVector& viewFactorRowSum() const{ return viewFactorRowSum_ ; }

        // Returns the per-triangle row-sum vector of the computed view-factor matrix for modification
        LinearVector& viewFactorRowSum() { return viewFactorRowSum_ ; }

        // Returns the maximum view-factor row sum across all emitting triangles
        double maxViewFactorRowSum(){ return maxViewFactorRowSum_ ; }

        // Returns the minimum view-factor row sum across all emitting triangles
        double minViewFactorRowSum(){ return minViewFactorRowSum_ ; }

        // Returns the view factor from triangle i to triangle j
        double operator()(int i, int j) const{ return viewFactors_(i,j);}

        // Returns the host-side triangle tracing history in read-only form
        const vector<TriangleTraceHistory>& h_triangleHistoryOut() const{ return this->h_triangleHistoryOut_ ; }

        // Returns the host-side triangle tracing history for modification
        vector<TriangleTraceHistory>& h_triangleHistoryOut() {return this->h_triangleHistoryOut_ ; }

        // Packs an emitting and receiving triangle index into a single 64-bit key
        uint64_t makeKey(int i, int j) const{ return (uint64_t(uint32_t(i)) << 32) | uint32_t(j); }

        // Extracts the emitting-triangle row index from a packed hit-count key
        int keyRow(uint64_t key) const{ return int(key >> 32);}

        // Extracts the receiving-triangle column index from a packed hit-count key
        int keyCol(uint64_t key) const{ return int(key & 0xffffffffu);}

        // Copies the OptiX tracing results from the launch manager into host-side view-factor data structures
        void getOptixSimulationResult(LaunchManager& launchManager, cudaStream_t& stream) ;

        // Accumulates emitted-ray counts and hit counts from one traced ray batch
        void accumulateBatch(LaunchManager& launchManager, cudaStream_t& stream) ;

        // Converts accumulated hit statistics into final sparse view factors and writes diagnostics
        void finaliseViewFactors(ofstream& out) ;

} ;