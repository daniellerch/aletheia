#include <iostream>
#include <math.h>
#include <cmath>

#ifndef CONFIG_H_
#define CONFIG_H_

class Config
{
public:
	bool verbose;
	int T;
	int order;
	int *SPAMsymmCoord;
	int SPAMsymmDim;
	int *MINMAXsymmCoord;
	int MINMAXsymmDim;
	bool mergeSpams;
	bool eraseLSB;
	bool parity;

	Config(bool verbose, int T, int order, bool symmSign, bool symmReverse, bool symmMinMax, bool mergeSpams, bool eraseLSB, bool parity)
	{
		this->verbose = verbose;
		this->T = T;
		this->order = order;
		this->mergeSpams = mergeSpams;
		this->eraseLSB = eraseLSB;
		this->parity = parity;
		
		GetSymmCoords(symmSign, symmReverse, symmMinMax);
	}

	~Config()
	{
		delete [] SPAMsymmCoord;
		delete [] MINMAXsymmCoord;
	}

private:

	void GetSymmCoords(bool symmSign, bool symmReverse, bool symmMinMax)
	{
		// Preparation of inSymCoord matrix for co-occurrence and symmetrization
		int B = 2*this->T+1;
		int fullDim = (int)std::pow((float)B, this->order);
	
		int alreadyUsed;
	
		// MINMAX
		alreadyUsed = 0;
		MINMAXsymmCoord = new int[2*fullDim]; // [0, fullDim-1] = min; [fullDim, 2*fullDim-1] = max
		for (int i=0; i<2*fullDim; i++) MINMAXsymmCoord[i] = -1;

		for (int numIter=0; numIter < fullDim; numIter++)
		{
			if (MINMAXsymmCoord[numIter] == -1)
			{
				int coordReverse = 0;
				int num = numIter;
				for (int i=0; i<this->order; i++)
				{
					coordReverse += (num % B) * ((int)std::pow((float)B, order-i-1));
					num = num / B;
				}
				// To the same bin: min(X), max(-X), min(Xreverse), max(-Xreverse)
				if (MINMAXsymmCoord[numIter] == -1)
				{
					MINMAXsymmCoord[numIter] = alreadyUsed; // min(X)
					if (symmMinMax) MINMAXsymmCoord[2*fullDim-numIter-1] = alreadyUsed; // max(-X)
					if (symmReverse) MINMAXsymmCoord[coordReverse] = alreadyUsed; // min(Xreverse)
					if ((symmMinMax) && (symmReverse)) MINMAXsymmCoord[2*fullDim-coordReverse-1] = alreadyUsed; // max(-Xreverse)
					alreadyUsed++;
				}
			}
		}
		for (int numIter=0; numIter < fullDim; numIter++)
		{
			if (MINMAXsymmCoord[fullDim+numIter] == -1)
			{
				int coordReverse = 0;
				int num = numIter;
				for (int i=0; i<this->order; i++)
				{
					coordReverse += (num % B) * ((int)std::pow((float)B, order-i-1));
					num = num / B;
				}
				// To the same bin: max(X), min(-X), max(Xreverse), min(-Xreverse)
				if (MINMAXsymmCoord[fullDim+numIter] == -1)
				{
					MINMAXsymmCoord[fullDim+numIter] = alreadyUsed; // max(X)
					if (symmMinMax) MINMAXsymmCoord[fullDim-numIter-1] = alreadyUsed; // min(-X)
					if (symmReverse) MINMAXsymmCoord[fullDim+coordReverse] = alreadyUsed; // max(Xreverse)
					if ((symmMinMax) && (symmReverse)) MINMAXsymmCoord[fullDim-coordReverse-1] = alreadyUsed; // min(-Xreverse)
					alreadyUsed++;
				}
			}
		}
		MINMAXsymmDim = alreadyUsed;
		
		// SPAM
		alreadyUsed = 0;
		SPAMsymmCoord = new int[fullDim];
		for (int i=0; i<fullDim; i++) SPAMsymmCoord[i] = -1;
		for (int numIter=0; numIter < fullDim; numIter++)
		{
			if (SPAMsymmCoord[numIter] == -1)
			{
				int coordReverse = 0;
				int num = numIter;
				for (int i=0; i<this->order; i++)
				{
					coordReverse += (num % B) * ((int)std::pow((float)B, order-i-1));
					num = num / B;
				}
				// To the same bin: X, -X, Xreverse, -Xreverse
				SPAMsymmCoord[numIter] = alreadyUsed; // X
				if (symmSign) SPAMsymmCoord[fullDim-numIter-1] = alreadyUsed; // -X
				if (symmReverse) SPAMsymmCoord[coordReverse] = alreadyUsed; // Xreverse
				if ((symmSign) && (symmReverse)) SPAMsymmCoord[fullDim-coordReverse-1] = alreadyUsed; // -Xreverse
				alreadyUsed++;
			}
		}
		SPAMsymmDim = alreadyUsed;
		// In order to have the same order of the features as the matlab SRM - shift +1
		for (int i=0; i<fullDim; i++) 
		{
			if (SPAMsymmCoord[i]==alreadyUsed-1) SPAMsymmCoord[i]=0;
			else SPAMsymmCoord[i]++;
		}
	}
};

#endif