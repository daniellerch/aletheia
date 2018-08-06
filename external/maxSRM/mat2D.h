#ifndef MAT2D_H_
#define MAT2D_H_

#include <vector>
#include <iostream>

template <class T>
class mat2D
{
public:
	int rows;
	int cols;

	mat2D(int rows, int cols)
	{
		this->rows = rows;
		this->cols = cols;
		this->vect = std::vector<T>(rows*cols, 0);
	}

	~mat2D()
	{
	}

	/*
	T Read(int pos)
	{
		return this->vect[pos];
	}
	*/

	T Read(int row, int col)
	{
		return this->vect[row*cols+col];
	}
	/*
	void Write(int pos, T val)
	{
		this->vect[pos] = val;
	}
	*/

	void Write(int row, int col, T val)
	{
		this->vect[row*cols+col] = val;
	}

	void Print(int rowFrom, int rowTo, int colFrom, int colTo)
	{
		std::cout << "\n";
		for (int r=rowFrom; r<=rowTo; r++)
		{	
			for (int c=colFrom; c<=colTo; c++)
			{
				std::cout << this->Read(r, c) << " ";
			}
			std::cout << "\n";
		}
	}

	void Print()
	{
		Print(0, this->rows-1, 0, this->cols-1);
	}

private:
	std::vector<T> vect;
};

#endif
