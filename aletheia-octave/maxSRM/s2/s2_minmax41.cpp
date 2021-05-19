#include "../submodel.h"
#include "../mat2D.h"
#include "../config.cpp"

class s2_minmax41: public Submodel
{
public:
	s2_minmax41(float q, Config *config) : Submodel(q) 
	{
		this->modelName =  "s2_minmax41";
		this->minmax = true;

		this->coocDirs.push_back("horver");

		Initialize(config);
	}

	~s2_minmax41()
	{
	}

	virtual void ComputeFea(std::vector<mat2D<int> *> QResVect)
	{
		std::vector<std::vector<mat2D<int> *> > OpVect;

		// [0] - Horizontal, [1] - Vertical, [2] - Diagonal, [3] - Minor Diagonal

		// Horizontal + Vertical + Diagonal + Minor Diagonal
		std::vector<mat2D<int> *> HVDM = std::vector<mat2D<int> *>();
		HVDM.push_back(QResVect[0]);HVDM.push_back(QResVect[1]);HVDM.push_back(QResVect[2]);HVDM.push_back(QResVect[3]);
		OpVect.push_back(HVDM);

		this->AddFea(OpVect);
	}
};
