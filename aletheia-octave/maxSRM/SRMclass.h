#ifndef SRMCLASS_H_
#define SRMCLASS_H_

#include "submodel.h"
#include "image.h"
#include "mat2D.h"
#include <vector>
#include "s.h"
#include "config.cpp"

class SRMclass
{
public:
	std::vector<s *> submodelClasses;

	SRMclass(Config *config);
	~SRMclass();
	void ComputeFeatures(void);

	void ComputeFeatures(mat2D<int> * image, mat2D<double> * map);
	std::vector<Submodel *> GetSubmodels();

private:
	bool verbose;
	Config *config;
	std::vector<Submodel *> AddedMergesSpams;

	std::vector<Submodel *> PostProcessing(std::vector<Submodel *> submodels);
	mat2D<int> *GetParity(mat2D<int> *I);
};

#endif