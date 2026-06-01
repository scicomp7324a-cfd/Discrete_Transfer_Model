# pragma once 

#include <iostream>
#include <vector>
#include <array>
#include <fstream>
#include <string>
#include <sstream>
#include <map>
#include <unordered_map>

#include "core/Triangle.hpp"
#include "radiation/RayArrayBuilder.hpp"
#include "radiation/RayGenerator.hpp"

using namespace std ;

class DataBuilder{

    private:

        vector<TriangleRawData> trianglesA_ ;
        vector<TriangleRawData> trianglesB_ ;

        vector<Vec3> verticesA_ ;
        vector<Vec3> verticesB_ ;

        vector<Tri> indicesA_ ;
        vector<Tri> indicesB_ ;

        vector<Vec3> normalA_ ;
        vector<Vec3> normalB_ ;

        vector<float3> rayOrigins_ ;
        vector<float3> rayDirections_ ;

        vector<int> originSurfaceIds_ ;
        vector<int> originTriangleIds_ ;

        vector<TriangleProperties> triangleProperties_ ;

        vector<int> originGlobalTriangleIds_ ;

        int numTotalRays_  = 0 ;
        int thetaNum_  = 0;
        int psiNum_ = 0 ;

        int batchStartTriangle_;
        int batchEndTriangle_;

    public:

        // Reads the two input STL surfaces and property files, then builds geometry, indexing, normals, and triangle property data
        DataBuilder(vector<string> filenames, vector<string> propertyfiles, int thetaNum, int psiNum):thetaNum_(thetaNum),
                                                                                                      psiNum_(psiNum)
        {

            if(filenames.size() != 2){ 
                cerr << "Invalid file input. Exactly two stl files should be provided. First surface 1 and then surface 2." ;
                throw runtime_error("Invalid file input. Exactly two stl files should be provided. First surface 1 and then surface 2.") ;
            }

            this->readStl(filenames[0], this->trianglesA_) ;
            this->readStl(filenames[1], this->trianglesB_) ;

            this->constructVertexAndIndexArray(this->trianglesA_, this->verticesA_, this->indicesA_, this->normalA_) ;
            this->constructVertexAndIndexArray(this->trianglesB_, this->verticesB_, this->indicesB_, this->normalB_) ;

            this->setTriangleProperties(propertyfiles[0], this->trianglesA_) ;
            this->setTriangleProperties(propertyfiles[1], this->trianglesB_) ;

            //this->buildRayData(thetaNum, psiNum) ;

        }

        // Returns the raw STL triangle data for surface A in read-only form
        const vector<TriangleRawData>& trianglesA() const{ return this->trianglesA_ ; }

        // Returns the raw STL triangle data for surface A for modification
        vector<TriangleRawData>& trianglesA(){ return this->trianglesA_ ; }

        // Returns the raw STL triangle data for surface B in read-only form
        const vector<TriangleRawData>& trianglesB() const{ return this->trianglesB_ ; }

        // Returns the raw STL triangle data for surface B for modification
        vector<TriangleRawData>& trianglesB(){ return this->trianglesB_ ; }

        // Returns the unique vertex array for surface A in read-only form
        const vector<Vec3>& verticesA() const{ return this->verticesA_ ; }

        // Returns the unique vertex array for surface A for modification
        vector<Vec3>& verticesA(){ return this->verticesA_ ; }

        // Returns the unique vertex array for surface B in read-only form
        const vector<Vec3>& verticesB() const{ return this->verticesB_ ; }

        // Returns the unique vertex array for surface B for modification
        vector<Vec3>& verticesB(){ return this->verticesB_ ; }

        // Returns the triangle index array for surface A in read-only form
        const vector<Tri>& indicesA() const{ return this->indicesA_ ; }

        // Returns the triangle index array for surface A for modification
        vector<Tri>& indicesA(){ return this->indicesA_ ; }

        // Returns the triangle index array for surface B in read-only form
        const vector<Tri>& indicesB() const{ return this->indicesB_ ; }

        // Returns the triangle index array for surface B for modification
        vector<Tri>& indicesB(){ return this->indicesB_ ; }

        // Returns the triangle normal array for surface A in read-only form
        const vector<Vec3>& normalA() const{ return this->normalA_ ; }

        // Returns the triangle normal array for surface A for modification
        vector<Vec3>& normalA(){ return this->normalA_ ; }

        // Returns the triangle normal array for surface B in read-only form
        const vector<Vec3>& normalB() const{ return this->normalB_ ; }

        // Returns the triangle normal array for surface B for modification
        vector<Vec3>& normalB(){ return this->normalB_ ; }

        // Returns the complete ray-origin array in read-only form
        const vector<float3>& rayOrigins() const{ return this->rayOrigins_ ; }

        // Returns the complete ray-origin array for modification
        vector<float3>& rayOrigins(){ return this->rayOrigins_ ; }

        // Returns the complete ray-direction array in read-only form
        const vector<float3>& rayDirections() const{ return this->rayDirections_ ; }

        // Returns the complete ray-direction array for modification
        vector<float3>& rayDirections(){ return this->rayDirections_ ; }

        // Returns the surface ID associated with each generated ray in read-only form
        const vector<int>& originSurfaceIds() const{ return this->originSurfaceIds_ ; }

        // Returns the surface ID associated with each generated ray for modification
        vector<int>& originSurfaceIds(){ return this->originSurfaceIds_ ; }

        // Returns the local triangle ID associated with each generated ray in read-only form
        const vector<int>& originTriangleIds() const{ return this->originTriangleIds_ ; }

        // Returns the local triangle ID associated with each generated ray for modification
        vector<int>& originTriangleIds(){ return this->originTriangleIds_ ; }

        // Returns the physical and radiative properties assigned to all triangles in read-only form
        const vector<TriangleProperties>& triangleProperties() const{ return this->triangleProperties_ ; }

        // Returns the physical and radiative properties assigned to all triangles for modification
        vector<TriangleProperties>& triangleProperties() {return this->triangleProperties_ ;} 

        // Returns the global triangle ID associated with each generated ray in read-only form
        const vector<int>& originGlobalTriangleIds() const{ return this->originGlobalTriangleIds_ ; }

        // Returns the global triangle ID associated with each generated ray for modification
        vector<int>& originGlobalTriangleIds(){ return this->originGlobalTriangleIds_ ;}

        // Returns the total number of generated rays for modification
        int& numTotalRays(){ return this->numTotalRays_ ;}

        // Returns the number of polar ray divisions for modification
        int& thetaNum(){ return this->thetaNum_ ;}

        // Returns the number of azimuthal ray divisions for modification
        int& psiNum(){ return this->psiNum_ ;}

        // Returns the starting global triangle index of the current ray-generation batch
        int batchStartTriangle() const { return batchStartTriangle_; }

        // Returns the ending global triangle index of the current ray-generation batch
        int batchEndTriangle() const { return batchEndTriangle_; }

        // Reads an ASCII STL file into raw triangle records
        void readStl(string filename, vector<TriangleRawData>& triangles) ;

        // Converts raw STL triangles into unique vertices, triangle indices, and per-triangle normals
        void constructVertexAndIndexArray(vector<TriangleRawData>& triangles, vector<Vec3>& vertices, vector<Tri>& indices, vector<Vec3>& normal) ;

        // Builds ray arrays for all triangles on both surfaces
        void buildRayData(int thetaNum, int psiNum) ;

        // Computes the area of a triangular surface element
        double calculateArea(TriangleRawData triangle) ;

        // Reads surface property data and assigns area, temperature, emissivity, and surface ID to each triangle
        void setTriangleProperties(string filename, vector<TriangleRawData>& triangles) ;

        // Builds ray arrays only for the specified global triangle batch
        void buildRayDataForTriangleBatch(int batchStartTriangle, int batchEndTriangle);

        // Splits a string into whitespace-separated tokens
        vector<string> splitWhiteSpace(string s) ;

        // Removes whitespace from the beginning and end of a string
        string stripWhiteSpace(string s) ;

        // Converts a vertex coordinate into a rounded key for duplicate-vertex detection
        Vec3Key makeKey(const Vec3& v, float eps = 1e-6f) ;
} ;