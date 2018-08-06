#include "../submodel.h"
#include "../mat2D.h"
#include "../config.cpp"

class s3x3_minmax22h: public Submodel
{
public:
	s3x3_minmax22h(float q, Config *config) : Submodel(q) 
	{
		this->modelName =  "s3x3_minmax22h";
		this->minmax = true;

		this->coocDirs.push_back("hor");
		this->coocDirs.push_back("ver");

		Initialize(config);
	}

	~s3x3_minmax22h()
	{
	}

	virtual void ComputeFea(std::vector<mat2D<int> *> QResVect)
	{
		std::vector<std::vector<mat2D<int> *> > OpVect;

		// [0] - Right, [1] - Left, [2] - Up, [3] - Down
		// [4] - All

		// Up + Down
		std::vector<mat2D<int> *> RL = std::vector<mat2D<int> *>();
		RL.push_back(QResVect[2]);RL.push_back(QResVect[3]);
		OpVect.push_back(RL);

		// Right + Left
		std::vector<mat2D<int> *> UD = std::vector<mat2D<int> *>();
		UD.push_back(QResVect[0]);UD.push_back(QResVect[1]);
		OpVect.push_back(UD);

		this->AddFea(OpVect);
	}
};
