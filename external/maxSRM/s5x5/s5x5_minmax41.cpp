#include "../submodel.h"
#include "../mat2D.h"
#include "../config.cpp"

class s5x5_minmax41: public Submodel
{
public:
	s5x5_minmax41(float q, Config *config) : Submodel(q) 
	{
		this->modelName =  "s5x5_minmax41";
		this->minmax = true;

		this->coocDirs.push_back("horver");

		Initialize(config);
	}

	~s5x5_minmax41()
	{
	}

	virtual void ComputeFea(std::vector<mat2D<int> *> QResVect)
	{
		std::vector<std::vector<mat2D<int> *> > OpVect;

		// [0] - Right, [1] - Left, [2] - Up, [3] - Down
		// [4] - All

		// Twice the same, vertical and horizontal co-occurrence
		// Right Up Left Down
		std::vector<mat2D<int> *> RLUD = std::vector<mat2D<int> *>();
		RLUD.push_back(QResVect[0]);RLUD.push_back(QResVect[1]);RLUD.push_back(QResVect[2]);RLUD.push_back(QResVect[3]);
		OpVect.push_back(RLUD);

		this->AddFea(OpVect);
	}
};
