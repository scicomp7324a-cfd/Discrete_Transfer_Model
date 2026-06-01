# include <iostream>
# include <vector>
# include <array>
# include <fstream>
# include <string>
# include <sstream>
# include <map>
# include <unordered_map>
# include <thrust/device_vector.h>
# include <thrust/host_vector.h>
# include <thrust/sort.h>
# include <thrust/reduce.h>
# include <thrust/remove.h>
# include <thrust/transform.h>
# include <thrust/copy.h>
# include <thrust/execution_policy.h>
# include <thrust/iterator/constant_iterator.h>

# include "optix/LaunchManager.hpp"
# include "radiation/ViewFactor.hpp"


static __host__ __device__
uint64_t makeHitKeyDevice(int i, int j)
{
    return (uint64_t(uint32_t(i)) << 32) | uint32_t(j);
}

__global__
void buildHitKeysKernel(const HitRecord* records, uint64_t* keys, unsigned int* validFlags, unsigned int numRays)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if(tid >= numRays){
        return;
    }

    const HitRecord rec = records[tid];

    if(rec.emitter >= 0 && rec.receiver >= 0){
        keys[tid] = makeHitKeyDevice(rec.emitter, rec.receiver);
        validFlags[tid] = 1u;
    }
    else{
        keys[tid] = 0ull;
        validFlags[tid] = 0u;
    }
}

__global__
void accumulateEmitsKernel(const HitRecord* records, unsigned int* emits, unsigned int numRays)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if(tid >= numRays){
        return;
    }

    const int emitter = records[tid].emitter;

    if(emitter >= 0){
        atomicAdd(&emits[emitter], 1u);
    }
}

void ViewFactor::getOptixSimulationResult(LaunchManager& launchManager, cudaStream_t& stream){
    cudaMemcpy( this->h_triangleHistoryOut().data(), reinterpret_cast<void*>(launchManager.d_triangleInfo()), launchManager.numTotalTriangles()*sizeof(TriangleTraceHistory), cudaMemcpyDeviceToHost );
    cudaStreamSynchronize(stream);
}

struct IsValidFlag
{
    __host__ __device__
    bool operator()(const unsigned int x) const
    {
        return x != 0u;
    }
};

void ViewFactor::accumulateBatch(LaunchManager& launchManager, cudaStream_t& stream)
{
    const unsigned int numRays = launchManager.numRays();

    thrust::device_vector<uint64_t> d_keys(numRays);
    thrust::device_vector<unsigned int> d_valid(numRays);

    const int threads = 256;
    const int blocks = (numRays + threads - 1) / threads;

    buildHitKeysKernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const HitRecord*>(launchManager.d_hitRecords()),
        thrust::raw_pointer_cast(d_keys.data()),
        thrust::raw_pointer_cast(d_valid.data()),
        numRays
    );

    accumulateEmitsKernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const HitRecord*>(launchManager.d_hitRecords()),
        thrust::raw_pointer_cast(d_emits_.data()),
        numRays
    );

    cudaStreamSynchronize(stream);

    thrust::device_vector<uint64_t> d_validKeys(numRays);

    auto validEnd = thrust::copy_if(
        thrust::cuda::par.on(stream),
        d_keys.begin(),
        d_keys.end(),
        d_valid.begin(),
        d_validKeys.begin(),
        IsValidFlag()
    );

    d_validKeys.resize(validEnd - d_validKeys.begin());

    if(d_validKeys.empty()){
        return;
    }

    thrust::sort(
        thrust::cuda::par.on(stream),
        d_validKeys.begin(),
        d_validKeys.end()
    );

    thrust::device_vector<unsigned int> d_ones(d_validKeys.size(), 1u);
    thrust::device_vector<uint64_t> d_batchUniqueKeys(d_validKeys.size());
    thrust::device_vector<unsigned int> d_batchCounts(d_validKeys.size());

    auto reduceEnd = thrust::reduce_by_key(
        thrust::cuda::par.on(stream),
        d_validKeys.begin(),
        d_validKeys.end(),
        d_ones.begin(),
        d_batchUniqueKeys.begin(),
        d_batchCounts.begin()
    );

    const size_t batchUnique = reduceEnd.first - d_batchUniqueKeys.begin();

    d_batchUniqueKeys.resize(batchUnique);
    d_batchCounts.resize(batchUnique);

    thrust::host_vector<uint64_t> h_batchUniqueKeys = d_batchUniqueKeys;
    thrust::host_vector<unsigned int> h_batchCounts = d_batchCounts;

    for(size_t k = 0; k < batchUnique; ++k){
        hitCounts_[h_batchUniqueKeys[k]] += h_batchCounts[k];
    }
}

void ViewFactor::finaliseViewFactors(ofstream& out)
{
    thrust::host_vector<unsigned int> h_emits = d_emits_;
    this->emits_.assign(h_emits.begin(), h_emits.end());

    out << "Unique hit-count records: "
        << hitCounts_.size() << endl;

    this->viewFactors().entries().clear();
    this->viewFactors().entries().reserve(hitCounts_.size());

    size_t counter = 0;

    for(const auto& item : hitCounts_){

        const uint64_t key = item.first;
        const unsigned int hits = item.second;

        const int i = keyRow(key);
        const int j = keyCol(key);

        if(this->emits_[i] == 0){
            continue;
        }

        const double Fij =
            static_cast<double>(hits) /
            static_cast<double>(this->emits_[i]);

        this->viewFactors().entries().push_back({i, j, Fij});
        this->viewFactorRowSum_(i) += Fij ; 

        counter++;

        if(counter % 1000000 == 0){
            out << counter << " view-factor records finalised." << endl;
        }
    }

    maxViewFactorRowSum_ = this->viewFactorRowSum_.maxVal() ;
    minViewFactorRowSum_ = this->viewFactorRowSum_.minVal() ;

    out << "Final sparse view-factor nnz: " << this->viewFactors().nnz() << endl ;
}




