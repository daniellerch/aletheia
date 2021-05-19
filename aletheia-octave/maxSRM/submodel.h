#include <string>
#include "mat2D.h"
#include <vector>
#include "config.cpp"

#ifndef SUBMODEL_H_
#define SUBMODEL_H_

class Submodel {

public:
	int symmDim;
	float q;
	std::string mergeInto;
	std::string modelName;
	std::vector<float *> fea;
	mat2D<double> *map;

	Submodel(float q);
	~Submodel();

	std::string GetName();
	std::vector<float *> ReturnFea();
	virtual void ComputeFea(std::vector<mat2D<int> *> QResVect) = 0;

protected:
	bool minmax;
	std::vector<std::string> coocDirs;
	void AddFea(std::vector<std::vector<mat2D<int> *> > OpVect);

	void Initialize(Config *config);

private:
	int T;
	int order;
	int *inSymmCoord;
	int *multi;
	int fullDim;

	void GetMinMax(std::vector<mat2D<int> *> residuals, mat2D<int> *outMin, mat2D<int> *outMax);
    float* GetCooc(mat2D<int>*SPAM_Min_Residual, mat2D<int>*Max_Residual, std::string direction);
};

#endif
