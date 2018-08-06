#include "../submodel.h"
#include "../mat2D.h"
#include "../config.cpp"

class s3_minmax34: public Submodel
{
public:
	s3_minmax34(float q, Config *config) : Submodel(q) 
	{
		this->modelName =  "s3_minmax34";
		this->minmax = true;

		this->coocDirs.push_back("horver");
		this->coocDirs.push_back("horver");
		this->coocDirs.push_back("horver");
		this->coocDirs.push_back("horver");

		Initialize(config);
	}

	~s3_minmax34()
	{
	}

	virtual void ComputeFea(std::vector<mat2D<int> *> QResVect)
	{
		std::vector<std::vector<mat2D<int> *> > OpVect;

		// [0] - Right, [1] - Left, [2] - Up, [3] - Down
		// [4] - Right_Up, [5] - Right_Down, [6] - Left_Up, [7] - Left_Down

		// Twice the same, vertical and horizontal co-occurrence
		// Right - Right Up - Up
		std::vector<mat2D<int> *> RRUU = std::vector<mat2D<int> *>();
		RRUU.push_back(QResVect[0]);RRUU.push_back(QResVect[4]);RRUU.push_back(QResVect[2]);
		OpVect.push_back(RRUU);

		// Up - Left Up - Left
		std::vector<mat2D<int> *> ULUL = std::vector<mat2D<int> *>();
		ULUL.push_back(QResVect[2]);ULUL.push_back(QResVect[6]);ULUL.push_back(QResVect[1]);
		OpVect.push_back(ULUL);

		// Left - Left Down - Down
		std::vector<mat2D<int> *> LLDD = std::vector<mat2D<int> *>();
		LLDD.push_back(QResVect[1]);LLDD.push_back(QResVect[7]);LLDD.push_back(QResVect[3]);
		OpVect.push_back(LLDD);

		// Down - Right Down - Right
		std::vector<mat2D<int> *> DRDR = std::vector<mat2D<int> *>();
		DRDR.push_back(QResVect[3]);DRDR.push_back(QResVect[5]);DRDR.push_back(QResVect[0]);
		OpVect.push_back(DRDR);

		this->AddFea(OpVect);
	}
};
