/*
 This function outputs co-occurrences of ALL 3rd-order residuals
 listed in Figure 1 in our journal HUGO paper (version from June 14), 
 including the naming convention.
 */

#include "../mat2D.h"
#include "../submodel.h"
#include "../config.cpp"
#include "../s.h"

#include "s3x3_spam11.cpp"
#include "s3x3_spam14h.cpp"
#include "s3x3_spam14v.cpp"
#include "s3x3_minmax22h.cpp"
#include "s3x3_minmax22v.cpp"
#include "s3x3_minmax24.cpp"
#include "s3x3_minmax41.cpp"

class s3x3 : s
{
public:
	void CreateKernels()
	{
		mat2D<int> *temp;
		cutEdgesForParityBy = 1;

			// Right Kernel
		temp = new mat2D<int>(3, 3);
		temp->Write(0, 0, 0); temp->Write(0, 1, 2); temp->Write(0, 2,-1);
		temp->Write(1, 0, 0); temp->Write(1, 1,-4); temp->Write(1, 2, 2);
		temp->Write(2, 0, 0); temp->Write(2, 1, 2); temp->Write(2, 2,-1);
		kerR = temp;

		// Left Kernel
		temp = new mat2D<int>(3, 3);
		temp->Write(0, 0,-1); temp->Write(0, 1, 2); temp->Write(0, 2, 0);
		temp->Write(1, 0, 2); temp->Write(1, 1,-4); temp->Write(1, 2, 0);
		temp->Write(2, 0,-1); temp->Write(2, 1, 2); temp->Write(2, 2, 0);
		kerL = temp;

		// Up Kernel
		temp = new mat2D<int>(3, 3);
		temp->Write(0, 0,-1); temp->Write(0, 1, 2); temp->Write(0, 2,-1);
		temp->Write(1, 0, 2); temp->Write(1, 1,-4); temp->Write(1, 2, 2);
		temp->Write(2, 0, 0); temp->Write(2, 1, 0); temp->Write(2, 2, 0);
		kerU = temp;

		// Down Kernel
		temp = new mat2D<int>(3, 3);
		temp->Write(0, 0, 0); temp->Write(0, 1, 0); temp->Write(0, 2, 0);
		temp->Write(1, 0, 2); temp->Write(1, 1,-4); temp->Write(1, 2, 2);
		temp->Write(2, 0,-1); temp->Write(2, 1, 2); temp->Write(2, 2,-1);
		kerD = temp;

		// All Kernel
		temp = new mat2D<int>(3, 3);
		temp->Write(0, 0,-1); temp->Write(0, 1, 2); temp->Write(0, 2,-1);
		temp->Write(1, 0, 2); temp->Write(1, 1,-4); temp->Write(1, 2, 2);
		temp->Write(2, 0,-1); temp->Write(2, 1, 2); temp->Write(2, 2,-1);
		kerAll = temp;
	}

	s3x3(std::vector<float> qs, Config *config) : s(qs, config)
	{
		this->CreateKernels();
		quantMultiplier = 4;

		for (int qIndex=0; qIndex < (int)qs.size(); qIndex++)
		{
			float q = qs[qIndex];
			std::vector<Submodel *> submodelsForQ;
			
			submodelsForQ.push_back(new s3x3_spam11(q, config));
			submodelsForQ.push_back(new s3x3_spam14h(q, config));
			submodelsForQ.push_back(new s3x3_spam14v(q, config));
			submodelsForQ.push_back(new s3x3_minmax22h(q, config));
			submodelsForQ.push_back(new s3x3_minmax22v(q, config));
			submodelsForQ.push_back(new s3x3_minmax24(q, config));
			submodelsForQ.push_back(new s3x3_minmax41(q, config));

			this->submodels.push_back(submodelsForQ);
		}
	}

	~s3x3()
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

		mat2D<double> * pMap = new mat2D<double>(img->rows-2, img->cols-2);
		for (int i=0; i<img->rows-2; i++)
			for (int j=0; j<img->cols-2; j++)
				pMap->Write(i, j, map->Read(i+1, j+1));
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