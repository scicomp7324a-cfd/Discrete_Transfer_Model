# pragma once

#include <optix.h>
#include <cuda_runtime.h>
#include "core/RayGeometry.hpp"
#include "core/TriangleTraceHistory.hpp"

struct HitRecord{
    int emitter ;
    int receiver ;
};

struct Params
{
    OptixTraversableHandle handle;

    float3* rayOrigins ;
    float3* rayDirections ;

    int* originSurfaceIds ;
    int* originTriangleIds ;
    int* originGlobalTriangleIds ;
    int* surfaceTriangleOffsets ;

    RayGeometry* out ;
    TriangleTraceHistory* triangleHistory ;
    HitRecord* hitRecords ;

    unsigned int numRays ;
    unsigned int numTotalTriangles ;
    unsigned int numTrianglesA ;
    unsigned int numTrianglesB ;
    
};