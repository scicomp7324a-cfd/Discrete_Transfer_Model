# pragma once 

struct Vec3
{
    float x, y, z;
};

struct Tri
{
    unsigned int a, b, c;
};

struct TriangleRawData
{
    Vec3 normal, vertex1, vertex2, vertex3 ;
};

struct Vec3Key{
    int x, y, z ;

    bool operator<(const Vec3Key& other) const {
        if (x != other.x) return x < other.x;
        if (y != other.y) return y < other.y;
        return z < other.z;
    }
} ;

struct TriangleProperties
{
    int triangleId ;
    TriangleRawData triData ;
    double area ;
    double temperature ;
    double emissivity ;
} ;