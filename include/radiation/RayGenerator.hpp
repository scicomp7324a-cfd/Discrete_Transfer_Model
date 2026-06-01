# pragma once

#include <iostream>
#include <vector>
#include <array>
#include <cmath>
#include <vector_types.h>
#include <cuda_runtime.h>

using namespace std ;

class RayGenerator{

    private:

    vector<float3> triangleVertex_ ;
    int thetaNum_ ;
    int psiNum_ ;

    int numRays_ ;

    float3 n_ ;
    float3 s_ ;
    float3 t_ ;

    float3 centroid_ ;
    vector<float3> directions_ ;

    public:

    // Constructs a local ray generator for one triangle and immediately generates its hemispherical ray directions
    RayGenerator(vector<float3> triangleVertex, float3 normal, int thetaNum, int psiNum): triangleVertex_(triangleVertex),
                                                                           n_(normal),
                                                                           thetaNum_(thetaNum),
                                                                           psiNum_(psiNum)
    {
        this->numRays_ = thetaNum*psiNum ;
        this->calculateCentroid() ;
        this->calculateReferenceVector() ;
        this->calculatePerpendicularReferenceVector() ;
        this->generateRays() ;
    } 

    // Returns the number of rays generated for this triangle
    int numRays(){
        return this->numRays_ ;
    }

    // Returns the slightly offset triangle centroid in read-only form
    const float3& centroid() const{
        return this->centroid_ ;
    }

    // Returns the slightly offset triangle centroid for modification
    float3& centroid(){
        return this->centroid_ ;
    }

    // Returns the generated ray directions in read-only form
    const vector<float3>& directions() const{
        return this->directions_ ;
    }

    // Returns the generated ray directions for modification
    vector<float3>& directions(){
        return this->directions_ ;
    }

    // Computes the dot product of two 3D vectors
    float dot(float3 n, float3 s){
        return (n.x*s.x + n.y*s.y + n.z*s.z) ;
    }

    /*
    void calculateNormal(){

        if(triangleVertex_.size() != 3){
            std::cerr << "Triangle must contain exactly 3 vertices." ;
            return ;
        }

        float3 A = this->triangleVertex_[0] ;
        float3 B = this->triangleVertex_[1] ;
        float3 C = this->triangleVertex_[2] ;
        
        float x = (B.y - A.y)*(C.z - A.z) - (B.z - A.z)*(C.y - A.y) ;
        float y = (B.z - A.z)*(C.x - A.x) - (B.x - A.x)*(C.z - A.z) ;
        float z = (B.x - A.x)*(C.y - A.y) - (B.y - A.y)*(C.x - A.x) ;

        this->n_ = make_float3(x, y, z) ;
        float mag = sqrtf(this->n_.x*this->n_.x + this->n_.y*this->n_.y + this->n_.z*this->n_.z)  ;
        this->n_.x = this->n_.x/mag ;
        this->n_.y = this->n_.y/mag ;
        this->n_.z = this->n_.z/mag ;
    }
    */

    // Computes the triangle centroid and offsets it slightly along the surface normal to avoid self-intersection
    void calculateCentroid(){

        if(triangleVertex_.size() != 3){
            std::cerr << "Triangle must contain exactly 3 vertices." ;
            return ;
        }
        
        float3 A = this->triangleVertex_[0] ;
        float3 B = this->triangleVertex_[1] ;
        float3 C = this->triangleVertex_[2] ;

        float x = (A.x + B.x + C.x)/3.0 ;
        float y = (A.y + B.y + C.y)/3.0 ;
        float z = (A.z + B.z + C.z)/3.0 ;

        double eps = 1e-9f;

        this->centroid_ = make_float3(x + eps*this->n_.x, y + eps*this->n_.y, z + eps*this->n_.z) ;

    }

    // Builds the first tangential reference direction by projecting a vertex-to-centroid vector onto the triangle plane
    void calculateReferenceVector(){

        float3 A = this->triangleVertex_[0] ;
        float x = A.x - this->centroid_.x ;
        float y = A.y - this->centroid_.y ;
        float z = A.z - this->centroid_.z ;

        this->s_ = make_float3(x, y, z) ;
        this->s_.x = this->s_.x - this->dot(this->s_, this->n_) * this->n_.x;
        this->s_.y = this->s_.y - this->dot(this->s_, this->n_) * this->n_.y;
        this->s_.z = this->s_.z - this->dot(this->s_, this->n_) * this->n_.z;

        float mag = sqrtf(this->s_.x*this->s_.x + this->s_.y*this->s_.y + this->s_.z*this->s_.z)  ;
        this->s_.x = this->s_.x/mag ;
        this->s_.y = this->s_.y/mag ;
        this->s_.z = this->s_.z/mag ;

    }

    // Computes the cross product of two 3D vectors
    float3 crossProduct(float3 n, float3 s){
        float3 prod = make_float3(n.y*s.z - n.z*s.y, n.z*s.x - n.x*s.z, n.x*s.y - n.y*s.x) ;
        return prod ; 
    }

    // Builds the second tangential reference direction perpendicular to both the normal and the first tangent
    void calculatePerpendicularReferenceVector(){
        this->t_ = this->crossProduct(this->n_, this->s_) ;
        float mag = sqrtf(this->t_.x*this->t_.x + this->t_.y*this->t_.y + this->t_.z*this->t_.z)  ;
        this->t_.x = this->t_.x/mag ;
        this->t_.y = this->t_.y/mag ;
        this->t_.z = this->t_.z/mag ;
    }

    // Converts local polar and azimuthal angles into a global 3D ray direction
    float3 calculateDirection(double theta, double psi){

        float3 A ;
        A.x = cosf(theta)*n_.x + sinf(theta)*cosf(psi)*s_.x + sinf(theta)*sinf(psi)*t_.x ;
        A.y = cosf(theta)*n_.y + sinf(theta)*cosf(psi)*s_.y + sinf(theta)*sinf(psi)*t_.y ;
        A.z = cosf(theta)*n_.z + sinf(theta)*cosf(psi)*s_.z + sinf(theta)*sinf(psi)*t_.z ;

        return A ;
    }

    // Generates the cosine-weighted hemispherical set of ray directions over the triangle surface
    void generateRays(){

        vector<double> theta ;
        vector<double> psi ;

        double pi = M_PI ;
        double dpsi = (2*pi)/this->psiNum_ ;
        
        for(int i = 0; i < this->thetaNum_; ++i){
            float r = (i+0.5)/static_cast<float>(this->thetaNum_) ;
            float thetaVal = acosf(sqrtf(1-r)) ;
            theta.push_back(thetaVal) ;
        }

        for(int i = 0; i < this->psiNum_; ++i){
            psi.push_back(i*dpsi) ;
        }

        for(double th: theta){
            for(double ps: psi){
                float3 direction = this->calculateDirection(th, ps) ; 
                this->directions_.push_back(direction) ;
            }
        }
    }

} ;
