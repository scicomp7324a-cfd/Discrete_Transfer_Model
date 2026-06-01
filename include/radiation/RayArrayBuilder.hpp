# pragma once

#include <iostream>
#include <vector>
#include <array>
#include <cmath>
#include <vector_types.h>
#include <cuda_runtime.h>

#include "core/Triangle.hpp"
#include "radiation/RayGenerator.hpp"

using namespace std ;


class RayArrayBuilder{

    private:

        const vector<Vec3>& vertices_ ;
        const vector<Vec3>& normal_ ;
        const vector<Tri>& indices_ ;
        int thetaNum_ ;
        int psiNum_ ;
        int surfaceId_ ;
        int numRays_ ;

        vector<float3> rayOrigins_ ;
        vector<float3> rayDirections_ ;
        vector<int> originTriangleIds_ ;
        vector<int> originSurfaceIds_ ;
        vector<int> originGlobalTriangleIds_ ;

        int globalTriangleOffset_ ;
        int startTriangle_ ;
        int endTriangle_ ;
        
    public:

        // Builds all ray origins, directions, and origin-identification arrays for a selected triangle batch on one surface
        RayArrayBuilder(const vector<Vec3>& vertices, const vector<Vec3>& normal, const vector<Tri>& indices, const int thetaNum, const int psiNum, const int surfaceId, const int globalTriangleOffset, const int startTriangle, const int endTriangle):vertices_(vertices),
            normal_(normal),
            indices_(indices),
            thetaNum_(thetaNum),
            psiNum_(psiNum),
            surfaceId_(surfaceId),
            numRays_(0),
            globalTriangleOffset_(globalTriangleOffset),
            startTriangle_(startTriangle),
            endTriangle_(endTriangle)
        {
            this->buildRayArrays() ;
        }

        // Returns the generated ray-origin array in read-only form
        const vector<float3>& rayOrigins() const{
            return this->rayOrigins_ ;
        }

        // Returns the generated ray-origin array for modification
        vector<float3>& rayOrigins(){
            return this->rayOrigins_ ;
        }

        // Returns the generated ray-direction array in read-only form
        const vector<float3>& rayDirections() const{
            return this->rayDirections_ ;
        }

        // Returns the generated ray-direction array for modification
        vector<float3>& rayDirections(){
            return this->rayDirections_ ;
        }

        // Returns the local triangle index associated with each generated ray in read-only form
        const vector<int>& originTriangleIds() const{
            return this->originTriangleIds_ ;
        }

        // Returns the local triangle index associated with each generated ray for modification
        vector<int>& originTriangleIds(){
            return this->originTriangleIds_ ;
        }

        // Returns the surface identifier associated with each generated ray in read-only form
        const vector<int>& originSurfaceIds() const{
            return this->originSurfaceIds_ ;
        } 

        // Returns the surface identifier associated with each generated ray for modification
        vector<int>& originSurfaceIds(){
            return this->originSurfaceIds_ ;
        }

        // Returns the global triangle index associated with each generated ray in read-only form
        const vector<int>& originGlobalTriangleIds() const{
            return originGlobalTriangleIds_ ;
        }

        // Returns the global triangle index associated with each generated ray for modification
        vector<int>& originGlobalTriangleIds(){
            return originGlobalTriangleIds_ ;
        }

        // Returns the total number of rays generated for the selected triangle batch
        int numRays(){
            return this->numRays_ ;
        }

        // Generates ray origins, directions, local triangle IDs, global triangle IDs, and surface IDs for the selected triangle range
        void buildRayArrays(){

            for(int i = startTriangle_; i < endTriangle_; ++i){

                Tri index = this->indices_[i] ;
                vector<float3> vert = {{this->vertices_[index.a].x, this->vertices_[index.a].y, this->vertices_[index.a].z}, 
                                       {this->vertices_[index.b].x, this->vertices_[index.b].y, this->vertices_[index.b].z}, 
                                       {this->vertices_[index.c].x, this->vertices_[index.c].y, this->vertices_[index.c].z}};

                float3 n = {this->normal_[i].x, this->normal_[i].y, this->normal_[i].z} ;
                RayGenerator rayGen(vert, n, this->thetaNum_, this->psiNum_) ;
                this->numRays_ += rayGen.numRays() ;

                for(float3 direction: rayGen.directions()){
                    this->rayOrigins_.push_back(rayGen.centroid()) ;
                    this->rayDirections_.push_back(direction) ;
                    this->originTriangleIds_.push_back(i) ;
                    this->originGlobalTriangleIds_.push_back(this->globalTriangleOffset_ + i);
                    this->originSurfaceIds_.push_back(surfaceId_) ;
                }
            }
        }
} ;
