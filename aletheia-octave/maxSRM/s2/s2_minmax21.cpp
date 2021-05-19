#include "../submodel.h"
#include "../mat2D.h"
#include "../config.cpp"

class s2_minmax21: public Submodel
{
public:
	s2_minmax21(float q, Config *config) : Submodel(q) 
	{
		this->modelName =  "s2_minmax21";
		this->minmax = true;

		this->coocDirs.push_back("horver");

		Initialize(config);
	}

	~s2_minmax21()
	{
	}

	virtual void ComputeFea(std::vector<mat2D<int> *> QResVect)
	{
		std::vector<std::vector<mat2D<int> *> > OpVect;

		// [0] - Horizontal, [1] - Vertical, [2] - Diagonal, [3] - Minor Diagonal

		// Horizontal + Vertical
		std::vector<mat2D<int> *> HV = std::vector<mat2D<int> *>();
		HV.push_back(QResVect[0]);HV.push_back(QResVect[1]);
		OpVect.push_back(HV);

		this->AddFea(OpVect);
	}
};
