#include "ofMain.h"
//#include "ofApp.h"
#include "som.hpp"
#include "Node.hpp"
#include <iostream>
#include <cstdlib>
#include <ctime>


//========================================================================
int main( ){
    ofSetupOpenGL(1280,1280,OF_WINDOW);			// <-------- setup the GL contex

    ofRunApp(new ofApp());

    return 0;
}
