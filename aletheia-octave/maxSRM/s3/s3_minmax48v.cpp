#include "../submodel.h"
#include "../mat2D.h"
#include "../config.cpp"

class s3_minmax48v: public Submodel
{
public:
	s3_minmax48v(float q, Config *config) : Submodel(q) 
	{
		this->modelName =  "s3_minmax48v";
		this->minmax = true;

		this->coocDirs.push_back("ver");
		this->coocDirs.push_back("ver");
		this->coocDirs.push_back("ver");
		this->coocDirs.push_back("ver");
		this->coocDirs.push_back("hor");
		this->coocDirs.push_back("hor");
		this->coocDirs.push_back("hor");
		this->coocDirs.push_back("hor");

		Initialize(config);
	}

	~s3_minmax48v()
	{
	}

	virtual void ComputeFea(std::vector<mat2D<int> *> QResVect)
	{
		std::vector<std::vector<mat2D<int> *> > OpVect;

		// [0] - Right, [1] - Left, [2] - Up, [3] - Down
		// [4] - Right_Up, [5] - Right_Down, [6] - Left_Up, [7] - Left_Down

		// Right Right-Up Up Left-Up
		std::vector<mat2D<int> *> RRUULU = std::vector<mat2D<int> *>();
		RRUULU.push_back(QResVect[0]);RRUULU.push_back(QResVect[4]);RRUULU.push_back(QResVect[2]);RRUULU.push_back(QResVect[6]);
		OpVect.push_back(RRUULU);

		// Right-Up Up Left-Up Left
		std::vector<mat2D<int> *> RUULUL = std::vector<mat2D<int> *>();
		RUULUL.push_back(QResVect[4]);RUULUL.push_back(QResVect[2]);RUULUL.push_back(QResVect[6]);RUULUL.push_back(QResVect[1]);
		OpVect.push_back(RUULUL);

		// Right Right-Down Down Left-Down
		std::vector<mat2D<int> *> RRDDLD = std::vector<mat2D<int> *>();
		RRDDLD.push_back(QResVect[0]);RRDDLD.push_back(QResVect[5]);RRDDLD.push_back(QResVect[3]);RRDDLD.push_back(QResVect[7]);
		OpVect.push_back(RRDDLD);

		// Right-Down Down Left-Down Left
		std::vector<mat2D<int> *> RDDLDL = std::vector<mat2D<int> *>();
		RDDLDL.push_back(QResVect[5]);RDDLDL.push_back(QResVect[3]);RDDLDL.push_back(QResVect[7]);RDDLDL.push_back(QResVect[1]);
		OpVect.push_back(RDDLDL);

		// Up Right-Up Right Right-Down
		std::vector<mat2D<int> *> URURRD = std::vector<mat2D<int> *>();
		URURRD.push_back(QResVect[2]);URURRD.push_back(QResVect[4]);URURRD.push_back(QResVect[0]);URURRD.push_back(QResVect[5]);
		OpVect.push_back(URURRD);

		// Right-Up Right Right-Down Down
		std::vector<mat2D<int> *> RURRDD = std::vector<mat2D<int> *>();
		RURRDD.push_back(QResVect[4]);RURRDD.push_back(QResVect[0]);RURRDD.push_back(QResVect[5]);RURRDD.push_back(QResVect[3]);
		OpVect.push_back(RURRDD);

		// Up Left-Up Left Left-Down
		std::vector<mat2D<int> *> ULULLD = std::vector<mat2D<int> *>();
		ULULLD.push_back(QResVect[2]);ULULLD.push_back(QResVect[6]);ULULLD.push_back(QResVect[1]);ULULLD.push_back(QResVect[7]);
		OpVect.push_back(ULULLD);

		// Left-Up Left Left-Down Down
		std::vector<mat2D<int> *> LULLDD = std::vector<mat2D<int> *>();
		LULLDD.push_back(QResVect[6]);LULLDD.push_back(QResVect[1]);LULLDD.push_back(QResVect[7]);LULLDD.push_back(QResVect[3]);
		OpVect.push_back(LULLDD);

		this->AddFea(OpVect);
	}
};
