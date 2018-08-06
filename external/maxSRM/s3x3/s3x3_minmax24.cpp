#include "../submodel.h"
#include "../mat2D.h"
#include "../config.cpp"

class s3x3_minmax24: public Submodel
{
public:
	s3x3_minmax24(float q, Config *config) : Submodel(q) 
	{
		this->modelName =  "s3x3_minmax24";
		this->minmax = true;

		this->coocDirs.push_back("horver");
		this->coocDirs.push_back("horver");
		this->coocDirs.push_back("horver");
		this->coocDirs.push_back("horver");

		Initialize(config);
	}

	~s3x3_minmax24()
	{
	}

	virtual void ComputeFea(std::vector<mat2D<int> *> QResVect)
	{
		std::vector<std::vector<mat2D<int> *> > OpVect;

		// [0] - Right, [1] - Left, [2] - Up, [3] - Down
		// [4] - All

		// Twice the same, vertical and horizontal co-occurrence
		// Right Up
		std::vector<mat2D<int> *> RU = std::vector<mat2D<int> *>();
		RU.push_back(QResVect[0]);RU.push_back(QResVect[2]);
		OpVect.push_back(RU);

		// Right Down
		std::vector<mat2D<int> *> RD = std::vector<mat2D<int> *>();
		RD.push_back(QResVect[0]);RD.push_back(QResVect[3]);
		OpVect.push_back(RD);

		// Left Up
		std::vector<mat2D<int> *> LU = std::vector<mat2D<int> *>();
		LU.push_back(QResVect[1]);LU.push_back(QResVect[2]);
		OpVect.push_back(LU);

		// Left Down
		std::vector<mat2D<int> *> LD = std::vector<mat2D<int> *>();
		LD.push_back(QResVect[1]);LD.push_back(QResVect[3]);
		OpVect.push_back(LD);

		this->AddFea(OpVect);
	}
};
