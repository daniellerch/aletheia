#include "../submodel.h"
#include "../mat2D.h"
#include "../config.cpp"

class s3_minmax22v: public Submodel
{
public:
	s3_minmax22v(float q, Config *config) : Submodel(q) 
	{
		this->modelName =  "s3_minmax22v";
		this->minmax = true;

		this->coocDirs.push_back("ver");
		this->coocDirs.push_back("hor");

		Initialize(config);
	}

	~s3_minmax22v()
	{
	}

	virtual void ComputeFea(std::vector<mat2D<int> *> QResVect)
	{
		std::vector<std::vector<mat2D<int> *> > OpVect;

		// [0] - Right, [1] - Left, [2] - Up, [3] - Down
		// [4] - Right_Up, [5] - Right_Down, [6] - Left_Up, [7] - Left_Down

		// Right + Left
		std::vector<mat2D<int> *> RL = std::vector<mat2D<int> *>();
		RL.push_back(QResVect[0]);RL.push_back(QResVect[1]);
		OpVect.push_back(RL);

		// Up + Down
		std::vector<mat2D<int> *> UD = std::vector<mat2D<int> *>();
		UD.push_back(QResVect[2]);UD.push_back(QResVect[3]);
		OpVect.push_back(UD);

		this->AddFea(OpVect);
	}
};
