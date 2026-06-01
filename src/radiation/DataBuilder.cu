#include <iostream>
#include <vector>
#include <array>
#include <fstream>
#include <sstream>
#include <string>

#include "core/Triangle.hpp"
#include "radiation/RayArrayBuilder.hpp"
#include "radiation/RayGenerator.hpp"
#include "radiation/DataBuilder.hpp"

using namespace std ;


vector<string> DataBuilder::splitWhiteSpace(string s) {
    istringstream iss(s);
    vector<string> out;
    for (string tok; iss >> tok; ) out.push_back(tok);
    return out;
}

string DataBuilder::stripWhiteSpace(string s){
    auto first = s.find_first_not_of(" \t\r");
    if (first == string::npos) return ""; 
    auto last  = s.find_last_not_of(" \t\r");
    return s.substr(first, last - first + 1);
}

void DataBuilder::readStl(string filename, vector<TriangleRawData>& triangles){

    ifstream in(filename) ;
    if(!in){
        cerr << "Failed to open file" << endl ;
        throw std::runtime_error("Failed to open file") ;
    }

    string line ;
    vector<string> strvec ;

    while(getline(in,line)){

        if(line.empty()) continue ;
        if(line.find("normal") != string::npos){

            TriangleRawData triangle ;
            line = stripWhiteSpace(line) ;
            strvec = splitWhiteSpace(line) ;
            triangle.normal.x = stof(strvec[2]) ;
            triangle.normal.y = stof(strvec[3]) ;
            triangle.normal.z = stof(strvec[4]) ;

            while(getline(in,line)){

                if(line.find("endloop") != string::npos){
                    break ;
                }

                if(line.find("vertex") != string::npos){

                    line = stripWhiteSpace(line) ;
                    strvec = splitWhiteSpace(line) ;
                    triangle.vertex1.x = stof(strvec[1]) ;
                    triangle.vertex1.y = stof(strvec[2]) ;
                    triangle.vertex1.z = stof(strvec[3]) ;

                    getline(in,line) ;
                    line = stripWhiteSpace(line) ;
                    strvec = splitWhiteSpace(line) ;
                    triangle.vertex2.x = stof(strvec[1]) ;
                    triangle.vertex2.y = stof(strvec[2]) ;
                    triangle.vertex2.z = stof(strvec[3]) ;

                    getline(in, line) ;
                    line = stripWhiteSpace(line) ;
                    strvec = splitWhiteSpace(line) ;
                    triangle.vertex3.x = stof(strvec[1]) ;
                    triangle.vertex3.y = stof(strvec[2]) ;
                    triangle.vertex3.z = stof(strvec[3]) ;
                }
            }

            triangles.push_back(triangle) ;
        }
    }
}

void DataBuilder::constructVertexAndIndexArray(vector<TriangleRawData>& triangles, vector<Vec3>& vertices, vector<Tri>& indices, vector<Vec3>& normal){

    map<Vec3Key, int> vertexMap ;

    for(TriangleRawData triangle: triangles){

        Vec3 v1 = triangle.vertex1 ;
        Vec3 v2 = triangle.vertex2 ;
        Vec3 v3 = triangle.vertex3 ;
        Vec3 n  = triangle.normal ;

        Vec3Key vk1 = this->makeKey(v1) ;
        Vec3Key vk2 = this->makeKey(v2) ;
        Vec3Key vk3 = this->makeKey(v3) ;
        
        if(vertexMap.find(vk1) == vertexMap.end()){ 
            vertexMap[vk1] = vertexMap.size() ;
            vertices.push_back(v1) ;
        }

        if(vertexMap.find(vk2) == vertexMap.end()){ 
            vertexMap[vk2] = vertexMap.size() ;
            vertices.push_back(v2) ;
        }

        if(vertexMap.find(vk3) == vertexMap.end()){ 
            vertexMap[vk3] = vertexMap.size() ;
            vertices.push_back(v3) ;
        }

        Tri index ;
        index.a = vertexMap[vk1] ;
        index.b = vertexMap[vk2] ;
        index.c = vertexMap[vk3] ;

        indices.push_back(index) ;
        normal.push_back(n) ;
    }

}

/*
void DataBuilder::buildRayData(int thetaNum, int psiNum){

    int offsetA = 0 ;
    int offsetB = this->indicesA().size() ;

    RayArrayBuilder surfaceArrayA(this->verticesA(), this->normalA(), this->indicesA(), thetaNum, psiNum, 0, offsetA) ;
    RayArrayBuilder surfaceArrayB(this->verticesB(), this->normalB(), this->indicesB(), thetaNum, psiNum, 1, offsetB) ;

    this->rayOrigins() = surfaceArrayA.rayOrigins() ;
    this->rayOrigins().insert(this->rayOrigins().end(), surfaceArrayB.rayOrigins().begin(), surfaceArrayB.rayOrigins().end()) ;

    this->rayDirections() = surfaceArrayA.rayDirections() ;
    this->rayDirections().insert(this->rayDirections().end(), surfaceArrayB.rayDirections().begin() , surfaceArrayB.rayDirections().end());

    this->originTriangleIds() = surfaceArrayA.originTriangleIds() ;
    this->originTriangleIds().insert(this->originTriangleIds().end(), surfaceArrayB.originTriangleIds().begin() , surfaceArrayB.originTriangleIds().end()) ;

    this->originSurfaceIds() = surfaceArrayA.originSurfaceIds() ;
    this->originSurfaceIds().insert(this->originSurfaceIds().end(), surfaceArrayB.originSurfaceIds().begin(), surfaceArrayB.originSurfaceIds().end()) ;

    this->originGlobalTriangleIds() = surfaceArrayA.originGlobalTriangleIds() ;
    this->originGlobalTriangleIds().insert(this->originGlobalTriangleIds().end(), surfaceArrayB.originGlobalTriangleIds().begin(), surfaceArrayB.originGlobalTriangleIds().end()) ;

    this->numTotalRays() = surfaceArrayA.numRays() + surfaceArrayB.numRays() ;
    
}
*/

void DataBuilder::buildRayDataForTriangleBatch(int batchStartTriangle, int batchEndTriangle){

    this->rayOrigins().clear() ;
    this->rayDirections().clear() ;
    this->originTriangleIds().clear() ;
    this->originSurfaceIds().clear() ;
    this->originGlobalTriangleIds().clear() ;
    this->numTotalRays() = 0 ;

    int numA = static_cast<int>(this->indicesA().size());
    int numB = static_cast<int>(this->indicesB().size());
    int offsetA = 0 ;
    int offsetB = static_cast<int>(this->indicesA().size()) ;

    // Surface A global interval: [0, numA)

    // Batch interval: [batchStartTriangle, batchEndTriangle)

    int startA = std::max(batchStartTriangle, offsetA) - offsetA;
    int endA   = std::min(batchEndTriangle, offsetB) - offsetA;
    startA = std::max(0, startA);
    endA   = std::min(numA, endA);

    // Surface B global interval: [offsetB, offsetB + numB)

    int startB = std::max(batchStartTriangle, offsetB) - offsetB;
    int endB   = std::min(batchEndTriangle, offsetB + numB) - offsetB;
    startB = std::max(0, startB);
    endB   = std::min(numB, endB);

    cout << "A local range: " << startA << " -> " << endA << endl;
    cout << "B local range: " << startB << " -> " << endB << endl;

    if(startA < endA){
        RayArrayBuilder surfaceArrayA(this->verticesA(), this->normalA(), this->indicesA(), this->thetaNum(), this->psiNum(), 0, offsetA, startA, endA) ;
        
        this->rayOrigins().insert(this->rayOrigins().end(), surfaceArrayA.rayOrigins().begin(), surfaceArrayA.rayOrigins().end()) ;
        this->rayDirections().insert(this->rayDirections().end(), surfaceArrayA.rayDirections().begin() , surfaceArrayA.rayDirections().end());
        this->originTriangleIds().insert(this->originTriangleIds().end(), surfaceArrayA.originTriangleIds().begin() , surfaceArrayA.originTriangleIds().end()) ;
        this->originSurfaceIds().insert(this->originSurfaceIds().end(), surfaceArrayA.originSurfaceIds().begin(), surfaceArrayA.originSurfaceIds().end()) ;
        this->originGlobalTriangleIds().insert(this->originGlobalTriangleIds().end(), surfaceArrayA.originGlobalTriangleIds().begin(), surfaceArrayA.originGlobalTriangleIds().end()) ;
        this->numTotalRays() += surfaceArrayA.numRays() ;
    }

    if(startB < endB){
        RayArrayBuilder surfaceArrayB(this->verticesB(), this->normalB(), this->indicesB(), this->thetaNum(), this->psiNum(), 1, offsetB, startB, endB) ;
        
        this->rayOrigins().insert(this->rayOrigins().end(), surfaceArrayB.rayOrigins().begin(), surfaceArrayB.rayOrigins().end()) ;
        this->rayDirections().insert(this->rayDirections().end(), surfaceArrayB.rayDirections().begin() , surfaceArrayB.rayDirections().end());
        this->originTriangleIds().insert(this->originTriangleIds().end(), surfaceArrayB.originTriangleIds().begin() , surfaceArrayB.originTriangleIds().end()) ;
        this->originSurfaceIds().insert(this->originSurfaceIds().end(), surfaceArrayB.originSurfaceIds().begin(), surfaceArrayB.originSurfaceIds().end()) ;
        this->originGlobalTriangleIds().insert(this->originGlobalTriangleIds().end(), surfaceArrayB.originGlobalTriangleIds().begin(), surfaceArrayB.originGlobalTriangleIds().end()) ;
        this->numTotalRays() += surfaceArrayB.numRays() ;
    }
}

double DataBuilder::calculateArea(TriangleRawData triangle){

    double a = (triangle.vertex2.y - triangle.vertex1.y)*(triangle.vertex3.z - triangle.vertex1.z) - (triangle.vertex2.z - triangle.vertex1.z)*(triangle.vertex3.y - triangle.vertex1.y) ;
    double b = (triangle.vertex2.z - triangle.vertex1.z)*(triangle.vertex3.x - triangle.vertex1.x) - (triangle.vertex2.x - triangle.vertex1.x)*(triangle.vertex3.z - triangle.vertex1.z) ;
    double c = (triangle.vertex2.x - triangle.vertex1.x)*(triangle.vertex3.y - triangle.vertex1.y) - (triangle.vertex2.y - triangle.vertex1.y)*(triangle.vertex3.x - triangle.vertex1.x) ;
    double area = 0.5*sqrt(a*a + b*b + c*c) ;
    return area ;

}

void DataBuilder::setTriangleProperties(string filename, vector<TriangleRawData>& triangles){

    ifstream in(filename) ;
    if(!in){
        cerr << "Failed to open file" << endl ;
        throw std::runtime_error("Failed to open file") ;
    }

    string line ;
    getline(in, line) ;
    int i = 0 ;

    while(getline(in, line)){

        stringstream ss(line) ;
        string cell ;
        vector<string> strvec ;

        while(getline(ss, cell, ',')){
            strvec.push_back(cell) ;
        }

        TriangleProperties tp ;
        tp.triData = triangles[i] ;
        tp.area = this->calculateArea(triangles[i]) ;
        tp.triangleId = this->triangleProperties().size() ;
        tp.temperature = stod(strvec[4]) ;
        tp.emissivity = stod(strvec[5]) ;

        this->triangleProperties().push_back(tp) ;
        ++i ;
    }
}

Vec3Key DataBuilder::makeKey(const Vec3& v, float eps){
    return{
        static_cast<int>(round(v.x / eps)),
        static_cast<int>(round(v.y / eps)),
        static_cast<int>(round(v.z/ eps))
    } ;
}




