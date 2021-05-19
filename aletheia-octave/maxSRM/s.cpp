#include "mat2D.h"
#include "submodel.h"
#include "image.h"
#include "s.h"
#include "config.cpp"
#include <vector>
#include <math.h>
#include <cmath>

s::s(std::vector<float> qs, Config *config)
{
	this->qs = qs;
	this->config = config;
}

s::~s()
{
	for (int i=0; i<(int)this->submodels.size(); i++) 
		for (int j=0; j<(int)this->submodels[i].size(); j++)
			delete submodels[i][j];
}

mat2D<int>* s::GetResidual(mat2D<int> *img, mat2D<int>* kernel)
{
	mat2D<int> *residual = new mat2D<int>(img->rows-kernel->rows+1, img->cols-kernel->cols+1);
	for (int ir=0; ir < (img->rows - kernel->rows + 1); ir++)
	{
		for (int ic=0; ic < (img->cols - kernel->cols + 1); ic++)
		{
			int convVal = 0;
			for (int kr=0; kr < kernel->rows; kr++)
			{
				for (int kc=0; kc < kernel->cols; kc++)
				{
					convVal = convVal + img->Read(ir+kr, ic+kc) * kernel->Read(kr, kc);
				}
			}
			residual->Write(ir, ic, convVal);
		}
	}
	return residual;
}

mat2D<int>* s::Quantize(mat2D<int>* residual, float totalKernelQ)
{
	mat2D<int>* quantizedResidual = new mat2D<int>(residual->rows, residual->cols);
	for (int r=0; r<residual->rows; r++)
	{
		for (int c=0; c<residual->cols; c++)
		{
			float tempValF = (float)residual->Read(r, c);
			tempValF = tempValF / totalKernelQ;
			int tempValI = 0;
			// rounding
			if (tempValF-floor(tempValF) > 0.5) tempValI = (int)ceil(tempValF);
			else tempValI = (int)floor(tempValF);
			if (tempValI > config->T) tempValI = config->T;
			if (tempValI < -config->T) tempValI = -config->T;
			quantizedResidual->Write(r, c, tempValI);
		}
	}
	return quantizedResidual;
}

void s::MultiplyByParity(std::vector<mat2D<int> *> QResVect, mat2D<int> *parity)
{
	for (int resIndex=0; resIndex < (int)QResVect.size(); resIndex++)
	{
		mat2D<int> *currentQRes = QResVect[resIndex];
		for (int r=0; r < currentQRes->rows; r++)
			for (int c=0; c < currentQRes->cols; c++)
			{
				int iValue = currentQRes->Read(r, c) * parity->Read(r + this->cutEdgesForParityBy, c + this->cutEdgesForParityBy);
				currentQRes->Write(r, c, iValue);
			}
	}
}
