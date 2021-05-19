/* 
 This class computes and contains co-occurrences of ALL 2nd-order residuals
 listed in Figure 1 in our journal HUGO paper (version from June 14), 
 including the naming convention.

 List of outputted features:

 1a) spam12h
 1b) spam12v (orthogonal-spam)
 1c) minmax21
 1d) minmax41
 1e) minmax24h (24v is also outputted but not listed in Figure 1)
 1f) minmax32

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

#include "s2_spam12h.cpp"
#include "s2_spam12v.cpp"
#include "s2_minmax21.cpp"
#include "s2_minmax24h.cpp"
#include "s2_minmax24v.cpp"
#include "s2_minmax32.cpp"
#include "s2_minmax41.cpp"

class s2 : s
{
public:
	void CreateKernels()
	{
		mat2D<int> *temp;
		cutEdgesForParityBy = 1;
		
		// Horizontal Kernel
		temp = new mat2D<int>(3, 3);
		temp->Write(0, 0, 0); temp->Write(0, 1, 0); temp->Write(0, 2, 0);
		temp->Write(1, 0, 1); temp->Write(1, 1,-2); temp->Write(1, 2, 1);
		temp->Write(2, 0, 0); temp->Write(2, 1, 0); temp->Write(2, 2, 0);
		kerH = temp;

		// Vertical Kernel
		temp = new mat2D<int>(3, 3);
		temp->Write(0, 0, 0); temp->Write(0, 1, 1); temp->Write(0, 2, 0);
		temp->Write(1, 0, 0); temp->Write(1, 1,-2); temp->Write(1, 2, 0);
		temp->Write(2, 0, 0); temp->Write(2, 1, 1); temp->Write(2, 2, 0);
		kerV = temp;

		// Diagonal Kernel
		temp = new mat2D<int>(3, 3);
		temp->Write(0, 0, 1); temp->Write(0, 1, 0); temp->Write(0, 2, 0);
		temp->Write(1, 0, 0); temp->Write(1, 1,-2); temp->Write(1, 2, 0);
		temp->Write(2, 0, 0); temp->Write(2, 1, 0); temp->Write(2, 2, 1);
		kerD = temp;

		// Minor diagonal Kernel
		temp = new mat2D<int>(3, 3);
		temp->Write(0, 0, 0); temp->Write(0, 1, 0); temp->Write(0, 2, 1);
		temp->Write(1, 0, 0); temp->Write(1, 1,-2); temp->Write(1, 2, 0);
		temp->Write(2, 0, 1); temp->Write(2, 1, 0); temp->Write(2, 2, 0);
		kerM = temp;
	}

	s2(std::vector<float> qs, Config *config) : s(qs, config)
	{
		this->CreateKernels();
		quantMultiplier = 2;

		for (int qIndex=0; qIndex < (int)qs.size(); qIndex++)
		{
			float q = qs[qIndex];
			std::vector<Submodel *> submodelsForQ;

			submodelsForQ.push_back(new s2_spam12h(q, config));
			submodelsForQ.push_back(new s2_spam12v(q, config));
			submodelsForQ.push_back(new s2_minmax21(q, config));
			submodelsForQ.push_back(new s2_minmax24h(q, config));
			submodelsForQ.push_back(new s2_minmax24v(q, config));
			submodelsForQ.push_back(new s2_minmax32(q, config));
			submodelsForQ.push_back(new s2_minmax41(q, config));

			this->submodels.push_back(submodelsForQ);
		}
	}

	~s2()
	{
		delete kerH; delete kerV; delete kerD; delete kerM;
	}

	void ComputeImage(mat2D<int> *img, mat2D<double> * map, mat2D<int> *parity)
	{
		mat2D<int> *H = GetResidual(img, kerH);
		mat2D<int> *V = GetResidual(img, kerV);
		mat2D<int> *D = GetResidual(img, kerD);
		mat2D<int> *M = GetResidual(img, kerM);

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
			QResVect.push_back(Quantize(H, q));
			QResVect.push_back(Quantize(V, q));
			QResVect.push_back(Quantize(D, q));
			QResVect.push_back(Quantize(M, q));


			// If parity is turned on
			if (config->parity) MultiplyByParity(QResVect, parity);

			for (int submodelIndex=0; submodelIndex < (int)submodels[qIndex].size(); submodelIndex++)
			{
				submodels[qIndex][submodelIndex]->ComputeFea(QResVect);
			}

			for (int i=0; i<(int)QResVect.size(); i++) delete QResVect[i];
		}
		delete H; delete V;delete D; delete M;
		delete pMap;
	}

private:
	mat2D<int> *kerH;
	mat2D<int> *kerV;
	mat2D<int> *kerD;
	mat2D<int> *kerM;
};