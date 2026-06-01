#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>

#include <optix.h>
#include <optix_stubs.h>
#include <optix_device.h>

#include "core/RayGeometry.hpp"
#include "core/RayParams.hpp"
#include "core/TriangleTraceHistory.hpp"

extern "C" {
    __constant__ Params params;
}

//static __forceinline__ __device__ int getGlobalTriangleIdLegacy(int surfaceId, int localTriangleId)
//{
//    if (surfaceId == 0) { return localTriangleId; }
//    return params.numTrianglesA + localTriangleId;
//}

extern "C" __global__ void __miss__ms()
{
    const unsigned int i = optixGetPayload_0() ;
    if(i >= params.numRays) return ;
    RayGeometry& geom = params.out[i] ;

    const float3 rayOrigin = optixGetWorldRayOrigin() ;
    const float3 rayDirection = optixGetWorldRayDirection() ; 

    geom.setHit(0) ;

    geom.setRayEnd(NAN, NAN, NAN) ;
    geom.setRayEndDirection(rayDirection.x, rayDirection.y, rayDirection.z) ;

    geom.setRayEndSurfaceId(-1) ;
    geom.setRayEndTriangleId(-1) ;

    geom.setDistance(-1.0f) ;

    const int emitterGlobal = params.originGlobalTriangleIds[i] ;
    params.hitRecords[i].emitter = emitterGlobal;
    params.hitRecords[i].receiver = -1;

    atomicAdd(&(params.triangleHistory[emitterGlobal].numMissEmissions()), 1u);

}

extern "C" __global__ void __closesthit__ch()
{
    const unsigned int i = optixGetPayload_0();
    if(i >= params.numRays) return ;
    RayGeometry& geom = params.out[i] ;

    const float3 rayOrigin = optixGetWorldRayOrigin() ;
    const float3 rayDirection = optixGetWorldRayDirection() ; 
    const float t = optixGetRayTmax() ;

    geom.setHit(1) ;

    geom.setRayEnd(rayOrigin.x + t*rayDirection.x, rayOrigin.y + t*rayDirection.y, rayOrigin.z + t*rayDirection.z) ;
    geom.setRayEndDirection(rayDirection.x, rayDirection.y, rayDirection.z) ;

    geom.setRayEndTriangleId(static_cast<int>(optixGetPrimitiveIndex())) ;
    geom.setRayEndSurfaceId(static_cast<int>(optixGetInstanceId())) ;

    float dirLen = sqrt(rayDirection.x*rayDirection.x + rayDirection.y*rayDirection.y + rayDirection.z*rayDirection.z) ;
    geom.setDistance(t*dirLen) ;

    const int emitterGlobal  = params.originGlobalTriangleIds[i] ;
    const int receiverGlobal = params.surfaceTriangleOffsets[geom.getRayEndSurfaceId()] + geom.getRayEndTriangleId();

    if (geom.getRayOriginSurfaceId() == geom.getRayEndSurfaceId()) {
        atomicAdd(&(params.triangleHistory[emitterGlobal].numSelfHits()), 1u);
        params.hitRecords[i].emitter = emitterGlobal;
        params.hitRecords[i].receiver = -1;
    }
    else{
        atomicAdd(&(params.triangleHistory[emitterGlobal].numHitEmissions()), 1u);
        atomicAdd(&(params.triangleHistory[receiverGlobal].numReceptions()), 1u);
        params.hitRecords[i].emitter = emitterGlobal;
        params.hitRecords[i].receiver = receiverGlobal;
    }

    
}

extern "C" __global__ void __raygen__rg()
{
    const unsigned int i = optixGetLaunchIndex().x ;

    if(i >= params.numRays) return ;
     
    RayGeometry& geom = params.out[i] ;

    geom.setHit(0) ;
    geom.setRayOrigin( params.rayOrigins[i].x,
                       params.rayOrigins[i].y,
                       params.rayOrigins[i].z) ;

    geom.setRayStartDirection( params.rayDirections[i].x,
                               params.rayDirections[i].y,
                               params.rayDirections[i].z) ;

    geom.setRayEnd(0.0f, 0.0f, 0.0f) ;
    geom.setRayEndDirection(0.0f, 0.0f, 0.0f) ;

    geom.setRayOriginSurfaceId(params.originSurfaceIds[i]) ;
    geom.setRayOriginTriangleId(params.originTriangleIds[i]) ;

    geom.setRayEndSurfaceId(-1) ;
    geom.setRayEndTriangleId(-1) ;

    geom.setDistance(0) ;

    const int emitterGlobal = params.originGlobalTriangleIds[i] ;
    atomicAdd(&(params.triangleHistory[emitterGlobal].numEmits()), 1u);

    unsigned int payload0 = i;

    optixTrace(
        params.handle,
        params.rayOrigins[i],
        params.rayDirections[i],
        0.0f,
        1e20f,
        0.0f,
        255,
        OPTIX_RAY_FLAG_NONE,
        0,
        1,
        0,
        payload0
    );

}

