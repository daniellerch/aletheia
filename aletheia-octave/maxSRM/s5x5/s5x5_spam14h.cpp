#include "../submodel.h"
#include "../mat2D.h"
#include "../config.cpp"

class s5x5_spam14h: public Submodel
{
public:
	s5x5_spam14h(float q, Config *config) : Submodel(q) 
	{
		this->modelName = "s5x5_spam14h";
		this->mergeInto = "s5x5_spam14hv";
		this->minmax = false;

		this->coocDirs.push_back("hor");
		this->coocDirs.push_back("hor");
		this->coocDirs.push_back("ver");
		this->coocDirs.push_back("ver");

		Initialize(config);
	}

	~s5x5_spam14h()
	{
	}

	virtual void ComputeFea(std::vector<mat2D<int> *> QResVect)
	{
		std::vector<std::vector<mat2D<int> *> > OpVect;

		// [0] - Right, [1] - Left, [2] - Up, [3] - Down
		// [4] - All

		// Up
		std::vector<mat2D<int> *> U = std::vector<mat2D<int> *>();
		U.push_back(QResVect[2]);
		OpVect.push_back(U);

		// Down
		std::vector<mat2D<int> *> D = std::vector<mat2D<int> *>();
		D.push_back(QResVect[3]);
		OpVect.push_back(D);

		// Right
		std::vector<mat2D<int> *> R = std::vector<mat2D<int> *>();
		R.push_back(QResVect[0]);
		OpVect.push_back(R);

		// Left
		std::vector<mat2D<int> *> L = std::vector<mat2D<int> *>();
		L.push_back(QResVect[1]);
		OpVect.push_back(L);

		this->AddFea(OpVect);
	}
};
