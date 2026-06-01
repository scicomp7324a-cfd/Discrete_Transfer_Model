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

class GasBuilder{

    private:

        CUdeviceptr d_vertices_   = 0;
        CUdeviceptr d_indices_   = 0;
        CUdeviceptr d_tempBuffer_ = 0;
        CUdeviceptr d_gasBuffer_  = 0;

        unsigned int numVertices_  = 0;
        unsigned int numTriangles_ = 0;
        OptixTraversableHandle handle_ = 0;

    public: 
        GasBuilder(){}

        ~GasBuilder(){
            if (d_vertices_)  cudaFree(reinterpret_cast<void*>(d_vertices_));
            if (d_indices_)   cudaFree(reinterpret_cast<void*>(d_indices_));
            if (d_tempBuffer_) cudaFree(reinterpret_cast<void*>(d_tempBuffer_));
            if (d_gasBuffer_) cudaFree(reinterpret_cast<void*>(d_gasBuffer_));
        }

        CUdeviceptr& d_vertices(){
            return this->d_vertices_ ;
        }

        CUdeviceptr& d_indices(){
            return this->d_indices_ ;
        }

        CUdeviceptr& d_tempBuffer(){
            return this->d_tempBuffer_ ;
        }

        CUdeviceptr& d_gasBuffer(){
            return this->d_gasBuffer_ ;
        }

        unsigned int& numVertices(){
            return this->numVertices_ ;
        }

        unsigned int& numTriangles(){
            return this->numTriangles_ ; 
        }

        OptixTraversableHandle& handle(){
            return this->handle_ ;
        }

        const OptixTraversableHandle& handle() const { 
            return handle_; 
        }

        void buildTriangleGas( OptixDeviceContext context, cudaStream_t stream, const std::vector<Vec3>& vertices, const std::vector<Tri>& indices);

} ;