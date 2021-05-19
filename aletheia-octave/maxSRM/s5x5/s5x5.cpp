/*
 This function outputs co-occurrences of ALL 3rd-order residuals
 listed in Figure 1 in our journal HUGO paper (version from June 14), 
 including the naming convention.
 */

#include "../mat2D.h"
#include "../submodel.h"
#include "../config.cpp"
#include "../s.h"

#include "s5x5_spam11.cpp"
#include "s5x5_spam14h.cpp"
#include "s5x5_spam14v.cpp"
#include "s5x5_minmax22h.cpp"
#include "s5x5_minmax22v.cpp"
#include "s5x5_minmax24.cpp"
#include "s5x5_minmax41.cpp"

class s5x5 : s
{
public:
	void CreateKernels()
	{
		mat2D<int> *temp;
		cutEdgesForParityBy = 2;

		// Right Kernel
		temp = new mat2D<int>(5, 5);
		temp->Write(0, 0, 0); temp->Write(0, 1, 0); temp->Write(0, 2, -2); temp->Write(0, 3, 2); temp->Write(0, 4,-1);
		temp->Write(1, 0, 0); temp->Write(1, 1, 0); temp->Write(1, 2,  8); temp->Write(1, 3,-6); temp->Write(1, 4, 2);
		temp->Write(2, 0, 0); temp->Write(2, 1, 0); temp->Write(2, 2,-12); temp->Write(2, 3, 8); temp->Write(2, 4,-2);
		temp->Write(3, 0, 0); temp->Write(3, 1, 0); temp->Write(3, 2,  8); temp->Write(3, 3,-6); temp->Write(3, 4, 2);
		temp->Write(4, 0, 0); temp->Write(4, 1, 0); temp->Write(4, 2, -2); temp->Write(4, 3, 2); temp->Write(4, 4,-1);
		kerR = temp;

		// Left Kernel
		temp = new mat2D<int>(5, 5);
		temp->Write(0, 0,-1); temp->Write(0, 1, 2); temp->Write(0, 2, -2); temp->Write(0, 3, 0); temp->Write(0, 4, 0);
		temp->Write(1, 0, 2); temp->Write(1, 1,-6); temp->Write(1, 2,  8); temp->Write(1, 3, 0); temp->Write(1, 4, 0);
		temp->Write(2, 0,-2); temp->Write(2, 1, 8); temp->Write(2, 2,-12); temp->Write(2, 3, 0); temp->Write(2, 4, 0);
		temp->Write(3, 0, 2); temp->Write(3, 1,-6); temp->Write(3, 2,  8); temp->Write(3, 3, 0); temp->Write(3, 4, 0);
		temp->Write(4, 0,-1); temp->Write(4, 1, 2); temp->Write(4, 2, -2); temp->Write(4, 3, 0); temp->Write(4, 4, 0);
		kerL = temp;

		// Up Kernel
		temp = new mat2D<int>(5, 5);
		temp->Write(0, 0,-1); temp->Write(0, 1, 2); temp->Write(0, 2, -2); temp->Write(0, 3, 2); temp->Write(0, 4,-1);
		temp->Write(1, 0, 2); temp->Write(1, 1,-6); temp->Write(1, 2,  8); temp->Write(1, 3,-6); temp->Write(1, 4, 2);
		temp->Write(2, 0,-2); temp->Write(2, 1, 8); temp->Write(2, 2,-12); temp->Write(2, 3, 8); temp->Write(2, 4,-2);
		temp->Write(3, 0, 0); temp->Write(3, 1, 0); temp->Write(3, 2,  0); temp->Write(3, 3, 0); temp->Write(3, 4, 0);
		temp->Write(4, 0, 0); temp->Write(4, 1, 0); temp->Write(4, 2,  0); temp->Write(4, 3, 0); temp->Write(4, 4, 0);
		kerU = temp;

		// Down Kernel
		temp = new mat2D<int>(5, 5);
		temp->Write(0, 0, 0); temp->Write(0, 1, 0); temp->Write(0, 2,  0); temp->Write(0, 3, 0); temp->Write(0, 4, 0);
		temp->Write(1, 0, 0); temp->Write(1, 1, 0); temp->Write(1, 2,  0); temp->Write(1, 3, 0); temp->Write(1, 4, 0);
		temp->Write(2, 0,-2); temp->Write(2, 1, 8); temp->Write(2, 2,-12); temp->Write(2, 3, 8); temp->Write(2, 4,-2);
		temp->Write(3, 0, 2); temp->Write(3, 1,-6); temp->Write(3, 2,  8); temp->Write(3, 3,-6); temp->Write(3, 4, 2);
		temp->Write(4, 0,-1); temp->Write(4, 1, 2); temp->Write(4, 2, -2); temp->Write(4, 3, 2); temp->Write(4, 4,-1);
		kerD = temp;

		// All Kernel
		temp = new mat2D<int>(5, 5);
		temp->Write(0, 0,-1); temp->Write(0, 1, 2); temp->Write(0, 2, -2); temp->Write(0, 3, 2); temp->Write(0, 4,-1);
		temp->Write(1, 0, 2); temp->Write(1, 1,-6); temp->Write(1, 2,  8); temp->Write(1, 3,-6); temp->Write(1, 4, 2);
		temp->Write(2, 0,-2); temp->Write(2, 1, 8); temp->Write(2, 2,-12); temp->Write(2, 3, 8); temp->Write(2, 4,-2);
		temp->Write(3, 0, 2); temp->Write(3, 1,-6); temp->Write(3, 2,  8); temp->Write(3, 3,-6); temp->Write(3, 4, 2);
		temp->Write(4, 0,-1); temp->Write(4, 1, 2); temp->Write(4, 2, -2); temp->Write(4, 3, 2); temp->Write(4, 4,-1);
		kerAll = temp;
	}

	s5x5(std::vector<float> qs, Config *config) : s(qs, config)
	{
		this->CreateKernels();
		quantMultiplier = 12;

		for (int qIndex=0; qIndex < (int)qs.size(); qIndex++)
		{
			float q = qs[qIndex];
			std::vector<Submodel *> submodelsForQ;

			submodelsForQ.push_back(new s5x5_spam11(q, config));
			submodelsForQ.push_back(new s5x5_spam14h(q, config));
			submodelsForQ.push_back(new s5x5_spam14v(q, config));
			submodelsForQ.push_back(new s5x5_minmax22h(q, config));
			submodelsForQ.push_back(new s5x5_minmax22v(q, config));
			submodelsForQ.push_back(new s5x5_minmax24(q, config));
			submodelsForQ.push_back(new s5x5_minmax41(q, config));

			this->submodels.push_back(submodelsForQ);
		}
	}

	~s5x5()
	{
		delete kerR; delete kerL; delete kerU; delete kerD;
		delete kerAll;
	}

	void ComputeImage(mat2D<int> *img, mat2D<double> * map, mat2D<int> *parity)
	{
		mat2D<int> *R = GetResidual(img, kerR);
		mat2D<int> *L = GetResidual(img, kerL);
		mat2D<int> *U = GetResidual(img, kerU);
		mat2D<int> *D = GetResidual(img, kerD);
		mat2D<int> *All = GetResidual(img, kerAll);

		mat2D<double> * pMap = new mat2D<double>(img->rows-4, img->cols-4);
		for (int i=0; i<img->rows-4; i++)
			for (int j=0; j<img->cols-4; j++)
				pMap->Write(i, j, map->Read(i+2, j+2));
		for (int i=0; i<this->submodels.size(); i++)
			for (int j=0; j<this->submodels[i].size(); j++)
				this->submodels[i][j]->map = pMap;

		for (int qIndex=0; qIndex < (int)submodels.size(); qIndex++)
		{
			float q = qs[qIndex] * quantMultiplier;
			std::vector<mat2D<int> *> QResVect;
			QResVect.push_back(Quantize(R, q));
			QResVect.push_back(Quantize(L, q));
			QResVect.push_back(Quantize(U, q));
			QResVect.push_back(Quantize(D, q));
			QResVect.push_back(Quantize(All, q));

			// If parity is turned on
			if (config->parity) MultiplyByParity(QResVect, parity);

			for (int submodelIndex=0; submodelIndex < (int)submodels[qIndex].size(); submodelIndex++)
			{
				// Compute the features for current submodel
				submodels[qIndex][submodelIndex]->ComputeFea(QResVect);
			}

			for (int i=0; i<(int)QResVect.size(); i++) delete QResVect[i];
		}
		delete R; delete L;delete U; delete D; 
		delete All;
		delete pMap;
	}

private:
	mat2D<int> *kerR;
	mat2D<int> *kerL;
	mat2D<int> *kerU;
	mat2D<int> *kerD;
	mat2D<int> *kerAll;
};