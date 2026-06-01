# pragma once

struct vec3{
    float x, y, z ;
} ;

class RayGeometry{

    private:

        int hit_ ;

        vec3 rayOrigin_ ;
        vec3 rayEnd_ ;

        vec3 rayStartDirection_ ;
        vec3 rayEndDirection_ ;

        int rayOriginSurfaceId_ ;
        int rayOriginTriangleId_ ;

        int rayEndSurfaceId_ ;
        int rayEndTriangleId_ ;
        
        float distance_ ;

    public:

        __host__ __device__ RayGeometry()
        : hit_(0),
          rayOrigin_{0.0f, 0.0f, 0.0f},
          rayEnd_{0.0f, 0.0f, 0.0f},
          rayStartDirection_{0.0f, 0.0f, 0.0f},
          rayEndDirection_{0.0f, 0.0f, 0.0f},
          rayOriginTriangleId_(-1),
          rayEndTriangleId_(-1),
          rayOriginSurfaceId_(-1),
          rayEndSurfaceId_(-1),
          distance_(0.0f){}

        __host__ __device__ void setHit(int hit){ this->hit_ = hit ;}
        __host__ __device__ int getHit() const{ return this->hit_ ;}

        __host__ __device__ void setRayOrigin(float x, float y, float z){
            rayOrigin_.x = x ;
            rayOrigin_.y = y ;
            rayOrigin_.z = z ;
        }
        __host__ __device__ vec3 getRayOrigin() const{return rayOrigin_ ;}

        __host__ __device__ void setRayEnd(float x, float y, float z){
            rayEnd_.x = x ;
            rayEnd_.y = y ;
            rayEnd_.z = z ;
        }
        __host__ __device__ vec3 getRayEnd() const{return rayEnd_ ;}

        __host__ __device__ void setRayStartDirection(float x, float y, float z){
            rayStartDirection_.x = x ;
            rayStartDirection_.y = y ;
            rayStartDirection_.z = z ;
        }
        __host__ __device__ vec3 getRayStartDirection() const{return rayStartDirection_ ;}

        __host__ __device__ void setRayEndDirection(float x, float y, float z){
            rayEndDirection_.x = x ;
            rayEndDirection_.y = y ;
            rayEndDirection_.z = z ;
        }
        __host__ __device__ vec3 getRayEndDirection() const{return rayEndDirection_ ;}

        __host__ __device__ void setRayOriginTriangleId(int rayOriginTriangleId){this->rayOriginTriangleId_ = rayOriginTriangleId ;}
        __host__ __device__ int getRayOriginTriangleId() const{return this->rayOriginTriangleId_ ;}
        
        __host__ __device__ void setRayEndTriangleId(int rayEndTriangleId){this->rayEndTriangleId_ = rayEndTriangleId ;}
        __host__ __device__ int getRayEndTriangleId() const{return this->rayEndTriangleId_ ;} 

        __host__ __device__ void setRayOriginSurfaceId(int rayOriginSurfaceId){this->rayOriginSurfaceId_ = rayOriginSurfaceId ;}
        __host__ __device__ int getRayOriginSurfaceId() const{return this->rayOriginSurfaceId_;}

        __host__ __device__ void setRayEndSurfaceId(int rayEndSurfaceId){this->rayEndSurfaceId_ = rayEndSurfaceId ;}
        __host__ __device__ int getRayEndSurfaceId() const{return this->rayEndSurfaceId_;}

        __host__ __device__ void setDistance(float distance){this->distance_ = distance ;}
        __host__ __device__ float getDistance() const{return this->distance_ ;} 

} ;