#include <vector>
#include "submodel.h"
#include "SRMclass.h"
#include <mex.h>
#include <cstring>

/*
	prhs[0] - cell array of image paths
	prhs[1] - struct config
				config.T			- int32		- default 2		- residual threshold
				config.order		- int32		- default 4		- co-occurrence order
				config.merge_spams	- logical	- default true	- if true then spam features are merged
				config.symm_sign	- logical	- default true	- if true then spam symmetry is used
				config.symm_reverse	- logical	- default true	- if true then reverse symmetry is used
				config.symm_minmax	- logical	- default true	- if true then minmax symmetry is used
				config.eraseLSB		- logical	- default false	- if true then all LSB are erased from the image
				config.parity		- logical	- default false	- if true then parity residual is applied
*/
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) 
{
	const mxArray *image = prhs[0];
	const mxArray *map = prhs[1];
	const mxArray *configStruct = prhs[2];

	// Default config
	int T = 2;
	int order = 4;
	bool mergeSpams = true;
	bool ss = true, sr = true, sm = true;
	bool eraseLSB = false, parity = false;

	if ((nrhs != 2) && (nrhs != 3))
	{
		mexErrMsgTxt ("Two or three inputs are required.\n2 inputs - [uint8 image] [double map]\n3 inputs - [uint8 image] [double map] [struct config]");
	}
	if  (!(mxIsClass(image, "uint8")))
	{
		mexErrMsgTxt ("The first input (cover image) must be a 'uint8' matrix.");
	}
	if  (!(mxIsClass(map, "double")))
	{
		mexErrMsgTxt ("The second input (probability map) must be a 'double' matrix.");
	}
	if (nrhs == 3)
	{
		int nfields = mxGetNumberOfFields(configStruct);
		if (nfields==0) mexErrMsgTxt ("The config structure is empty.");
		for(int fieldIndex=0; fieldIndex<nfields; fieldIndex++)
		{
			const char *fieldName = mxGetFieldNameByNumber(configStruct, fieldIndex);
			const mxArray *fieldContent = mxGetFieldByNumber(configStruct, 0, fieldIndex);
			// if a field is not scalar
			if ((mxGetM(fieldContent)!= 1) || (mxGetN(fieldContent)!= 1))
				mexErrMsgTxt ("All config fields must be scalars.");
			// if every field is scalar
			if (strcmp(fieldName, "T") == 0)
				if (mxIsClass(fieldContent, "int32")) T = (int)mxGetScalar(fieldContent);
				else mexErrMsgTxt ("'config.T' must be of type 'int32'");
			if (strcmp(fieldName, "order") == 0)
				if (mxIsClass(fieldContent, "int32")) order = (int)mxGetScalar(fieldContent);
				else mexErrMsgTxt ("'config.order' must be of type 'int32'");
			if (strcmp(fieldName, "merge_spams") == 0)
				if (mxIsLogical(fieldContent)) mergeSpams = mxIsLogicalScalarTrue(fieldContent);
				else mexErrMsgTxt ("'config.mergeSpams' must be of type 'logical'");
			if (strcmp(fieldName, "symm_sign") == 0)
				if (mxIsLogical(fieldContent)) ss = mxIsLogicalScalarTrue(fieldContent);
				else mexErrMsgTxt ("'config.symm_sign' must be of type 'logical'");
			if (strcmp(fieldName, "symm_reverse") == 0)
				if (mxIsLogical(fieldContent)) sr = mxIsLogicalScalarTrue(fieldContent);
				else mexErrMsgTxt ("'config.symm_reverse' must be of type 'logical'");
			if (strcmp(fieldName, "symm_minmax") == 0)
				if (mxIsLogical(fieldContent)) sm = mxIsLogicalScalarTrue(fieldContent);
				else mexErrMsgTxt ("'config.symm_minmax' must be of type 'logical'");
			if (strcmp(fieldName, "eraseLSB") == 0)
				if (mxIsLogical(fieldContent)) eraseLSB = mxIsLogicalScalarTrue(fieldContent);
				else mexErrMsgTxt ("'config.eraseLSB' must be of type 'logical'");
			if (strcmp(fieldName, "parity") == 0)
				if (mxIsLogical(fieldContent)) parity = mxIsLogicalScalarTrue(fieldContent);
				else mexErrMsgTxt ("'config.parity' must be of type 'logical'");
		}
	}

	// create C cover matrix
	int rows = (int)mxGetM(image);
	int cols = (int)mxGetN(image);
	mat2D<int> *c_image = new mat2D<int>(rows, cols);
	unsigned char *image_array = (unsigned char *)mxGetData(image);

	for (int c=0; c<cols; c++)
	{
		for (int r=0; r<rows; r++)
		{
			c_image->Write(r, c, (int)image_array[r+c*rows]);
		}
	}

	// create MAP cover matrix
	mat2D<double> *c_map = new mat2D<double>(rows, cols);
	double *map_array = (double *)mxGetData(map);

	for (int c=0; c<cols; c++)
	{
		for (int r=0; r<rows; r++)
		{
			c_map->Write(r, c, map_array[r+c*rows]);
		}
	}
	
	// create config object
	Config *config = new Config(false, T, order, ss, sr, sm, mergeSpams, eraseLSB, parity);

	// create object with all the submodels and compute the features
	SRMclass *SRMobj = new SRMclass(config);

	// Run the feature computation
	SRMobj->ComputeFeatures(c_image, c_map);

	delete c_image;
	delete c_map;

	std::vector<Submodel *> submodels = SRMobj->GetSubmodels();
	const char **submodelNames = new const char*[submodels.size()];
	for (int submodelIndex=0; submodelIndex < (int)submodels.size(); submodelIndex++) 
	{
		submodelNames[submodelIndex] = (new std::string(submodels[submodelIndex]->GetName()))->c_str();
	}
	mwSize structSize[2];
	structSize[0] = 1;
	structSize[1] = 1;
	plhs[0] = mxCreateStructArray(1, structSize, submodels.size(), submodelNames);
	for (int submodelIndex=0; submodelIndex < submodels.size(); submodelIndex++)
	{
		Submodel *currentSubmodel = submodels[submodelIndex];
		mwSize feaSize[2];
		feaSize[0] = (int)currentSubmodel->ReturnFea().size();
		feaSize[1] = currentSubmodel->symmDim;
		mxArray *fea = mxCreateNumericArray(2, feaSize, mxSINGLE_CLASS, mxREAL);
		for (int r=0; r<(int)currentSubmodel->ReturnFea().size(); r++)
		{
			for (int c=0; c<currentSubmodel->symmDim; c++)
			{
				((float*)mxGetPr(fea))[(c*(int)currentSubmodel->ReturnFea().size())+r]=(currentSubmodel->ReturnFea())[r][c];
			}
		}
		mxSetFieldByNumber(plhs[0],0,submodelIndex,fea);
	}

	delete SRMobj;
} 
