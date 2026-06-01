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
#include "optix/GasBuilder.hpp"
#include "core/RayGeometry.hpp"

using namespace std ;

class TriangleTraceHistory{

    private:

        int triangleId_ ;
        int surfaceId_ ;

        unsigned int numEmits_ ;
        unsigned int numReceptions_ ;
        unsigned int numHitEmissions_ ;
        unsigned int numMissEmissions_ ;
        unsigned int numSelfHits_ ;

    public:

        __host__ __device__ TriangleTraceHistory():triangleId_(-1),
                                                   surfaceId_(-1),
                                                   numEmits_(0),
                                                   numReceptions_(0),
                                                   numHitEmissions_(0),
                                                   numMissEmissions_(0),
                                                   numSelfHits_(0)
        {}

        __host__ __device__ TriangleTraceHistory(int triangleId, int surfaceId):triangleId_(triangleId),
                                                                                             surfaceId_(surfaceId),
                                                                                             numEmits_(0),
                                                                                             numReceptions_(0),
                                                                                             numHitEmissions_(0),
                                                                                             numMissEmissions_(0),
                                                                                             numSelfHits_(0)
        {}

        __host__ __device__ int triangleId() const{ return this->triangleId_ ; }
        __host__ __device__ int surfaceId() const{ return this->surfaceId_ ; }

        __host__ __device__ unsigned int& numEmits() { return this->numEmits_ ; }
        __host__ __device__ unsigned int& numReceptions() { return this->numReceptions_ ; }
        __host__ __device__ unsigned int& numHitEmissions() { return this->numHitEmissions_ ; }
        __host__ __device__ unsigned int& numMissEmissions() { return this->numMissEmissions_ ; }
        __host__ __device__ unsigned int& numSelfHits() { return this->numSelfHits_ ;}

        __host__ void setTriangleId(int triangleId){ this->triangleId_ = triangleId ; }
        __host__ void setSurfaceId(int surfaceId){ this->surfaceId_ = surfaceId ; }

    } ;