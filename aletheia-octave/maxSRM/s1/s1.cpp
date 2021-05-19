/*
 This class computes and contains co-occurrences of ALL 1st-order residuals
 listed in Figure 1 in our journal HUGO paper (version from June 14), 
 including the naming convention.

 List of outputted features:

 1a) spam14h
 1b) spam14v (orthogonal-spam)
 1c) minmax22v
 1d) minmax24
 1e) minmax34v
 1f) minmax41
 1g) minmax34
 1h) minmax48h
 1i) minmax54

 Naming convention:

 name = {type}{f}{sigma}{scan}
 type \in {spam, minmax}
 f \in {1,2,3,4,5} number of filters that are "minmaxed"
 sigma \in {1,2,3,4,8} symmetry index
 scan \in {h,v,\emptyset} scan of the cooc matrix (empty = sum of both 
 h and v scans).
*/

#include "../mat2D.h"
#include "../submodel.h"
#include "../config.cpp"
#include "../s.h"

#include "s1_spam14h.cpp"
#include "s1_spam14v.cpp"
#include "s1_minmax22h.cpp"
#include "s1_minmax22v.cpp"
#include "s1_minmax24.cpp"
#include "s1_minmax34.cpp"
#include "s1_minmax34h.cpp"
#include "s1_minmax34v.cpp"
#include "s1_minmax41.cpp"
#include "s1_minmax48h.cpp"
#include "s1_minmax48v.cpp"
#include "s1_minmax54.cpp"

class s1 : s
{
public:
	void CreateKernels()
	{
		mat2D<int> *temp;
		cutEdgesForParityBy = 1;
		
		// Right Kernel
		temp = new mat2D<int>(3, 3);
		temp->Write(0, 0, 0); temp->Write(0, 1, 0); temp->Write(0, 2, 0);
		temp->Write(1, 0, 0); temp->Write(1, 1,-1); temp->Write(1, 2, 1);
		temp->Write(2, 0, 0); temp->Write(2, 1, 0); temp->Write(2, 2, 0);
		kerR = temp;

		// Left Kernel
		temp = new mat2D<int>(3, 3);
		temp->Write(0, 0, 0); temp->Write(0, 1, 0); temp->Write(0, 2, 0);
		temp->Write(1, 0, 1); temp->Write(1, 1,-1); temp->Write(1, 2, 0);
		temp->Write(2, 0, 0); temp->Write(2, 1, 0); temp->Write(2, 2, 0);
		kerL = temp;

		// Up Kernel
		temp = new mat2D<int>(3, 3);
		temp->Write(0, 0, 0); temp->Write(0, 1, 1); temp->Write(0, 2, 0);
		temp->Write(1, 0, 0); temp->Write(1, 1,-1); temp->Write(1, 2, 0);
		temp->Write(2, 0, 0); temp->Write(2, 1, 0); temp->Write(2, 2, 0);
		kerU = temp;

		// Down Kernel
		temp = new mat2D<int>(3, 3);
		temp->Write(0, 0, 0); temp->Write(0, 1, 0); temp->Write(0, 2, 0);
		temp->Write(1, 0, 0); temp->Write(1, 1,-1); temp->Write(1, 2, 0);
		temp->Write(2, 0, 0); temp->Write(2, 1, 1); temp->Write(2, 2, 0);
		kerD = temp;

		// Right Up Kernel
		temp = new mat2D<int>(3, 3);
		temp->Write(0, 0, 0); temp->Write(0, 1, 0); temp->Write(0, 2, 1);
		temp->Write(1, 0, 0); temp->Write(1, 1,-1); temp->Write(1, 2, 0);
		temp->Write(2, 0, 0); temp->Write(2, 1, 0); temp->Write(2, 2, 0);
		kerRU = temp;

		// Right Down Kernel
		temp = new mat2D<int>(3, 3);
		temp->Write(0, 0, 0); temp->Write(0, 1, 0); temp->Write(0, 2, 0);
		temp->Write(1, 0, 0); temp->Write(1, 1,-1); temp->Write(1, 2, 0);
		temp->Write(2, 0, 0); temp->Write(2, 1, 0); temp->Write(2, 2, 1);
		kerRD = temp;

		// Left Up Kernel
		temp = new mat2D<int>(3, 3);
		temp->Write(0, 0, 1); temp->Write(0, 1, 0); temp->Write(0, 2, 0);
		temp->Write(1, 0, 0); temp->Write(1, 1,-1); temp->Write(1, 2, 0);
		temp->Write(2, 0, 0); temp->Write(2, 1, 0); temp->Write(2, 2, 0);
		kerLU = temp;

		// Left Down Kernel
		temp = new mat2D<int>(3, 3);
		temp->Write(0, 0, 0); temp->Write(0, 1, 0); temp->Write(0, 2, 0);
		temp->Write(1, 0, 0); temp->Write(1, 1,-1); temp->Write(1, 2, 0);
		temp->Write(2, 0, 1); temp->Write(2, 1, 0); temp->Write(2, 2, 0);
		kerLD = temp;
	}

	s1(std::vector<float> qs, Config *config) : s(qs, config)
	{
		this->CreateKernels();
		quantMultiplier = 1;

		for (int qIndex=0; qIndex < (int)qs.size(); qIndex++)
		{
			float q = qs[qIndex];
			std::vector<Submodel *> submodelsForQ;

			submodelsForQ.push_back(new s1_spam14h(q, config));
			submodelsForQ.push_back(new s1_spam14v(q, config));
			submodelsForQ.push_back(new s1_minmax22h(q, config));
			submodelsForQ.push_back(new s1_minmax22v(q, config));
			submodelsForQ.push_back(new s1_minmax24(q, config));
			submodelsForQ.push_back(new s1_minmax34(q, config));
			submodelsForQ.push_back(new s1_minmax34h(q, config));
			submodelsForQ.push_back(new s1_minmax34v(q, config));
			submodelsForQ.push_back(new s1_minmax41(q, config));
			submodelsForQ.push_back(new s1_minmax48h(q, config));
			submodelsForQ.push_back(new s1_minmax48v(q, config));
			submodelsForQ.push_back(new s1_minmax54(q, config));

			this->submodels.push_back(submodelsForQ);
		}
	}

	~s1()
	{
		delete kerR; delete kerL; delete kerU; delete kerD;
		delete kerRU; delete kerRD; delete kerLU; delete kerLD;
	}

	void ComputeImage(mat2D<int> *img, mat2D<double> * map, mat2D<int> *parity)
	{
		mat2D<int> *R = GetResidual(img, kerR);
		mat2D<int> *L = GetResidual(img, kerL);
		mat2D<int> *U = GetResidual(img, kerU);
		mat2D<int> *D = GetResidual(img, kerD);
		mat2D<int> *RU = GetResidual(img, kerRU);
		mat2D<int> *RD = GetResidual(img, kerRD);
		mat2D<int> *LU = GetResidual(img, kerLU);
		mat2D<int> *LD = GetResidual(img, kerLD);

		mat2D<double> * pMap = new mat2D<double>(R->rows, R->cols);
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
			QResVect.push_back(Quantize(RU, q));
			QResVect.push_back(Quantize(RD, q));
			QResVect.push_back(Quantize(LU, q));
			QResVect.push_back(Quantize(LD, q));

			// If parity is turned on
			if (config->parity) MultiplyByParity(QResVect, parity);

			for (int submodelIndex=0; submodelIndex < (int)submodels[qIndex].size(); submodelIndex++)
			{
				submodels[qIndex][submodelIndex]->ComputeFea(QResVect);
			}

			for (int i=0; i<(int)QResVect.size(); i++) delete QResVect[i];
		}
		delete R; delete L;delete U; delete D; 
		delete RU; delete RD; delete LU; delete LD;
		delete pMap;
	}

private:
	mat2D<int> *kerR;
	mat2D<int> *kerL;
	mat2D<int> *kerU;
	mat2D<int> *kerD;
	mat2D<int> *kerRU;
	mat2D<int> *kerRD;
	mat2D<int> *kerLU;
	mat2D<int> *kerLD;
};