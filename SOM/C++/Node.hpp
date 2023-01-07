#ifndef NODE 
#define NODE
#include <vector>

class Node
{

private:
	std::vector<double> weight;
	double x,y;

public:

	Node();

	//Node(int nb_weight, int i, int j);

    void initNodeWithColor();

    void initNodeWithRndColor();

    void initNodeWithCoord();

	void PrintNode();

	void PrintNodeVector();

    double getXofLattice();

    double getYofLattice();

    double getXCoord();

    double getYCoord();

	double GetDistance(std::vector<double> input);

	std::vector<double> GetVector();

	void initNodeCoor(int i, int j);

	void AjustWeight(Node n, double lr, double infl);

	~Node();

	
};
#endif
