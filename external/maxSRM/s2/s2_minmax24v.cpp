#include "../submodel.h"
#include "../mat2D.h"
#include "../config.cpp"

class s2_minmax24v: public Submodel
{
public:
	s2_minmax24v(float q, Config *config) : Submodel(q) 
	{
		this->modelName =  "s2_minmax24v";
		this->minmax = true;

		this->coocDirs.push_back("ver");
		this->coocDirs.push_back("ver");
		this->coocDirs.push_back("hor");
		this->coocDirs.push_back("hor");

		Initialize(config);
	}

	~s2_minmax24v()
	{
	}

	virtual void ComputeFea(std::vector<mat2D<int> *> QResVect)
	{
		std::vector<std::vector<mat2D<int> *> > OpVect;

		// [0] - Horizontal, [1] - Vertical, [2] - Diagonal, [3] - Minor Diagonal

		// Horizontal + Minor Diag
		std::vector<mat2D<int> *> HM = std::vector<mat2D<int> *>();
		HM.push_back(QResVect[0]);HM.push_back(QResVect[3]);
		OpVect.push_back(HM);

		// Horizontal + Diag
		std::vector<mat2D<int> *> HD = std::vector<mat2D<int> *>();
		HD.push_back(QResVect[0]);HD.push_back(QResVect[2]);
		OpVect.push_back(HD);

		// Vertical + Diag
		std::vector<mat2D<int> *> VD = std::vector<mat2D<int> *>();
		VD.push_back(QResVect[1]);VD.push_back(QResVect[2]);
		OpVect.push_back(VD);

		// Vertical + Minor Diag
		std::vector<mat2D<int> *> VM = std::vector<mat2D<int> *>();
		VM.push_back(QResVect[1]);VM.push_back(QResVect[3]);
		OpVect.push_back(VM);

		this->AddFea(OpVect);
	}
};
