#include "som.hpp"
#include "Node.hpp"
#include <vector>
#include <iostream>
#include <algorithm>
#include <cmath>

using namespace std;

Som::Som(int w, int h, int ep, int m)
{
    srand(time(NULL));
    epoch = ep;
    width = w;
    height = h;
    map_radius = max(width, height)/2;
    lambda = epoch/log(map_radius);
    learning_rate = start_lr;
    current_time = 0;
    mode = m;

    network = new Node* [height];
    for (int k = 0; k < height; ++k)
    {
        network [k] = new Node[width];
    }
    for (int i = 0; i < height; ++i)
    {
        for (int j = 0; j < width; ++j)
        {
            network [i][j].initNodeCoor(i,j);
            //network [i][j].initNodeWithCoord();
            if(mode == 0)
            {
                network [i][j].initNodeWithRndColor();
            }
            else
            {
                network[i][j].initNodeWithCoord();
            }
        }
    }
}

void Som::PrintSOM()
{
	for (int i = 0; i < height; ++i)
	{
            for (int j = 0; j < width; ++j)
            {
                network [i][j].PrintNodeVector();
            }
            cout<<endl;
	}
}

void Som::BestMatchUnit(Node n)
{
	float best = 1000;
	float last = 1000;
	int i_tmp, j_tmp;
	for (int i = 0; i < height; ++i)
	{
		for (int j = 0; j < width; ++j)
		{
			last = network [i][j].GetDistance(n.GetVector());
			if (last < best)
			{
				best = last;
				i_tmp = i;
				j_tmp = j;
			}
		}
	}

	bmu = network[i_tmp][j_tmp];
}

void Som::NeighbourRadius(int iteration_count)
{
    neighbour_rad = map_radius * exp(-(double)iteration_count/lambda);
}

void Som::CalculateNewWeights(Node example)
{
    for (int i = 0; i < height; ++i)
    {
        for (int j = 0; j < width; ++j)
        {
            double dist_node = (bmu.getXofLattice()-network[i][j].getXofLattice())*
                               (bmu.getXofLattice()-network[i][j].getXofLattice())+
                               (bmu.getYofLattice()-network[i][j].getYofLattice())*
                               (bmu.getYofLattice()-network[i][j].getYofLattice());

                double widthsq = neighbour_rad * neighbour_rad;
                if (dist_node < (neighbour_rad * neighbour_rad))
                {
                    influence = exp(-(dist_node) / (2*widthsq));
                    //influence = exp(-sqrt((dist_node)) / (2*100));

                    network[i][j].AjustWeight(example,learning_rate,influence);
                }
        }
    }
}

void Som::ReduceLR(int iteration_count)
{
    learning_rate = start_lr * exp(-(double)iteration_count/lambda);
}

int Som::GetWidth()
{
    return width;
}

int Som::GetHeight()
{
    return height;
}

Node Som::GetNodeAt(int i, int j)
{
    return network[i][j];
}

void Som::RunColorSOM()
{
    if(current_time < epoch)
    {
        float t = 0.0;
        Node n;
        n.initNodeWithColor();
        BestMatchUnit(n);
        CalculateNewWeights(n);
        NeighbourRadius(current_time);
        ReduceLR(current_time);
    }
    ++current_time;
    ofLog(OF_LOG_NOTICE) << current_time;
}

void Som::setup()
{
    ofBackground(0,0,0);
    //ofEnableSmoothing();
    ofEnableAlphaBlending();
    ofSetWindowTitle("Self Organising Map");

    ofSetRectMode(OF_RECTMODE_CENTER);

    ofSetFrameRate(60); // if vertical sync is off, we can go a bit fast... this caps the framerate at 60fps.
}

void Som::draw()
{
    if(mode == 0)
    {
        drawColor();
    }
    else
    {
        drawPoints();
    }
}

void Som::drawPoints()
{
    ofBackground(255);  // Clear the screen with a black color
    ofSetColor(0);
    ofDrawCircle(0, 1200, 10);

    for(int i = 0 ;i < height;++i)
    {
        for(int j = 0;j < width;++j)
        {
            ofDrawCircle(network[i][j].getXCoord(), network[i][j].getYCoord() , 10);
            if(j < width - 1)
            {
                ofDrawLine(network[i][j].getXCoord(), network[i][j].getYCoord(), network[i][j+1].getXCoord(), network[i][j+1].getYCoord());
            }
            if(i < height - 1)
            {
                ofDrawLine(network[i][j].getXCoord(), network[i][j].getYCoord(), network[i+1][j].getXCoord(), network[i+1][j].getYCoord());
            }
        }
    }
}

void Som::drawColor()
{
    int step = 20;
    int k,l = 0;

    for(int i = 0 ;i < height;++i)
    {
        for(int j = 0;j < width;++j)
        {
            ofColor c;
            std::vector <double> v = network[i][j].GetVector();
            c.r = v[0];
            c.g = v[1];
            c.b = v[2];
            ofSetColor(c);
            ofDrawRectangle(k,l,step-1,step-1);
            l = l + step;
        }
        l = 0;
        k = k + step;
    }
}

void Som::update()
{
    if(mode == 0)
    {
        RunColorSOM();
    }
    else
    {
        RunPointSOM();
    }
    //ofLog(OF_LOG_NOTICE) << current_time;

}

void Som::RunPointSOM()
{
    if(current_time < epoch)
    {
        float t = 0.0;
        Node n;
        n.initNodeWithCoord();
        BestMatchUnit(n);
        CalculateNewWeights(n);
        NeighbourRadius(current_time);
        ReduceLR(current_time);
    }
    ++current_time;
}

Som::~Som(){

	for(int i = 0; i<height; i++)
	{
    	delete [] network[i];
	}

	delete [] network;
}
