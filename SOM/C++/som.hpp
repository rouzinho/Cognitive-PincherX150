#ifndef SOM
#define SOM
#include <vector>
#include "Node.hpp"
#include "ofApp.h"

class Som
{

private:

    Node **network;
    int width, height;
    Node bmu;
    double map_radius;
    double lambda;
    double neighbour_rad;
    double influence;
    double learning_rate;
    int epoch;
    int current_time;
    const double start_lr = 0.1;
    int mode;

public:

    Som(int w, int h, int num_iter,int m);

    void PrintSOM();

    int GetWidth();

    int GetHeight();

    void BestMatchUnit(Node n);

    void CalculateNewWeights(Node example);

    void NeighbourRadius(int iteration_count);

    void ReduceLR(int num_iter);

    Node GetNodeAt(int i, int j);

    void RunColorSOM();

    void RunPointSOM();

    void setup();

    void update();

    void draw();

    void drawColor();

    void drawPoints();

	~Som();


};
#endif
