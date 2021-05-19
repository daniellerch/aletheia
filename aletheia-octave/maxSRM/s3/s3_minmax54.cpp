#include "../submodel.h"
#include "../mat2D.h"
#include "../config.cpp"

class s3_minmax54: public Submodel
{
public:
	s3_minmax54(float q, Config *config) : Submodel(q) 
	{
		this->modelName =  "s3_minmax54";
		this->minmax = true;

		this->coocDirs.push_back("horver");
		this->coocDirs.push_back("horver");
		this->coocDirs.push_back("horver");
		this->coocDirs.push_back("horver");

		Initialize(config);
	}

	~s3_minmax54()
	{
	}

	virtual void ComputeFea(std::vector<mat2D<int> *> QResVect)
	{
		std::vector<std::vector<mat2D<int> *> > OpVect;

		// [0] - Right, [1] - Left, [2] - Up, [3] - Down
		// [4] - Right_Up, [5] - Right_Down, [6] - Left_Up, [7] - Left_Down

		// Twice the same, vertical and horizontal co-occurrence
		// Right-Down Right Right-Up Up Left-Up
		std::vector<mat2D<int> *> RDRRUULU = std::vector<mat2D<int> *>();
		RDRRUULU.push_back(QResVect[5]);RDRRUULU.push_back(QResVect[0]);RDRRUULU.push_back(QResVect[4]);RDRRUULU.push_back(QResVect[2]);RDRRUULU.push_back(QResVect[6]);
		OpVect.push_back(RDRRUULU);

		// Right-Up Up Left-Up Left Left-Down
		std::vector<mat2D<int> *> RUULULLD = std::vector<mat2D<int> *>();
		RUULULLD.push_back(QResVect[4]);RUULULLD.push_back(QResVect[2]);RUULULLD.push_back(QResVect[6]);RUULULLD.push_back(QResVect[1]);RUULULLD.push_back(QResVect[7]);
		OpVect.push_back(RUULULLD);

		// Right-Up Right Right-Down Down Left-Down
		std::vector<mat2D<int> *> RURRDDLD = std::vector<mat2D<int> *>();
		RURRDDLD.push_back(QResVect[4]);RURRDDLD.push_back(QResVect[0]);RURRDDLD.push_back(QResVect[5]);RURRDDLD.push_back(QResVect[3]);RURRDDLD.push_back(QResVect[7]);
		OpVect.push_back(RURRDDLD);

		// Right-Down Down Left-Down Left Left-Up
		std::vector<mat2D<int> *> RDDLDLLU = std::vector<mat2D<int> *>();
		RDDLDLLU.push_back(QResVect[5]);RDDLDLLU.push_back(QResVect[3]);RDDLDLLU.push_back(QResVect[7]);RDDLDLLU.push_back(QResVect[1]);RDDLDLLU.push_back(QResVect[6]);
		OpVect.push_back(RDDLDLLU);

		this->AddFea(OpVect);
	}
};
