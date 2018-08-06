#include "../submodel.h"
#include "../mat2D.h"
#include "../config.cpp"

class s5x5_spam11: public Submodel
{
public:
	s5x5_spam11(float q, Config *config) : Submodel(q) 
	{
		this->modelName = "s5x5_spam11";
		this->mergeInto = "s35_spam11";
		this->minmax = false;

		this->coocDirs.push_back("horver");

		Initialize(config);
	}

	~s5x5_spam11()
	{
	}

	virtual void ComputeFea(std::vector<mat2D<int> *> QResVect)
	{
		std::vector<std::vector<mat2D<int> *> > OpVect;

		// [0] - Right, [1] - Left, [2] - Up, [3] - Down
		// [4] - All

		// All
		std::vector<mat2D<int> *> A = std::vector<mat2D<int> *>();
		A.push_back(QResVect[4]);
		OpVect.push_back(A);

		this->AddFea(OpVect);
	}
};
