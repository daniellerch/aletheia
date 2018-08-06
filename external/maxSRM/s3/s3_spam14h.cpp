#include "../submodel.h"
#include "../mat2D.h"
#include "../config.cpp"

class s3_spam14h: public Submodel
{
public:
	s3_spam14h(float q, Config *config) : Submodel(q) 
	{
		this->modelName =  "s3_spam14h";
		this->mergeInto = "s3_spam14hv";
		this->minmax = false;

		this->coocDirs.push_back("hor");
		this->coocDirs.push_back("ver");

		Initialize(config);
	}

	~s3_spam14h()
	{
	}

	virtual void ComputeFea(std::vector<mat2D<int> *> QResVect)
	{
		std::vector<std::vector<mat2D<int> *> > OpVect;

		// [0] - Right, [1] - Left, [2] - Up, [3] - Down
		// [4] - Right_Up, [5] - Right_Down, [6] - Left_Up, [7] - Left_Down

		// Right
		std::vector<mat2D<int> *> R = std::vector<mat2D<int> *>();
		R.push_back(QResVect[0]);
		OpVect.push_back(R);

		// Up
		std::vector<mat2D<int> *> U = std::vector<mat2D<int> *>();
		U.push_back(QResVect[2]);
		OpVect.push_back(U);

		this->AddFea(OpVect);
	}
};
