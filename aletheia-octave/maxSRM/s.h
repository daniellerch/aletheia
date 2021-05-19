#ifndef S_H_
#define S_H_

#include "mat2D.h"
#include "submodel.h"
#include "image.h"
#include "config.cpp"
#include <vector>

class s
{
public:
	std::vector<std::vector<Submodel *> > submodels;

	s(std::vector<float> qs, Config *config);
	~s();

	virtual void ComputeImage(mat2D<int> *img, mat2D<double> *map, mat2D<int> *parity) = 0;

protected:
	Config *config;
	std::vector<float> qs;
	int quantMultiplier;
	int cutEdgesForParityBy;

	mat2D<int> *GetResidual(mat2D<int> *img, mat2D<int>* kernel);
	mat2D<int> *Quantize(mat2D<int>* residual, float totalKernelQ);
	void MultiplyByParity(std::vector<mat2D<int> *> QResVect, mat2D<int> *parity);

};

#endif
