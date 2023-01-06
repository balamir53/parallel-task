#include <stdio.h>
#include <stdlib.h>
#include <string.h>
// #include <intrin.h>
#include <iostream>
#include <fstream>
#include "Vector2d.h"
#include <chrono>
#include <math.h>
// #include<bits/stdc++.h>
using namespace std;
using namespace std::chrono;

// function declarations //////////////////////////////////////////////////////
bool loadRawImage16(const char *fileName, int x, int y, unsigned short *data);
int* checkNeighbour(int* matrix, int* vectorV, int x, int y);
int (*(getFivetoFiveMatrix)(int* vector, int matrix[][5], int row, int column, int rowLength))[5];
int (*(addMatrix)(int* vector, int matrix[][5], int row, int column, int rowLength))[5];
//void freeMemory(int* array, int size);

// CONSTANTS //////////////////////////////////////////////////////////////////
const char *FILE_NAME = "/home/ziya/parallel-task/lena16bit.raw";
const int MAX_NAME = 512;               // max length of string
const int IMAGE_X = 256;                // image width
const int IMAGE_Y = 256;                // image height
const int apron = 2;
const int sizeOfVectorArray = (IMAGE_X + apron*2) * (IMAGE_X + apron * 2);
const float THRESHHOLD = 5;

// global variables ///////////////////////////////////////////////////////////
char    fileName[MAX_NAME];             // image file name
int     imageX, imageY;                 // image resolution

// image matrix
int* imageM;
unsigned short* readData;

int main(int argc, char **argv)
{
	// use default image file if not specified
	if (argc == 4)
	{
		strcpy(fileName, argv[1]);
		imageX = atoi(argv[2]);
		imageY = atoi(argv[3]);
	}
	else {
		printf("Usage: %s <image-file> <width> <height>\n", argv[0]);
		strcpy(fileName, FILE_NAME);
		imageX = IMAGE_X;
		imageY = IMAGE_Y;
		printf("\nUse default image \"%s\", (%d,%d)\n", fileName, imageX, imageY);
	}

	// allocate memory for image data, 16 bits per pixel
	readData = new unsigned short[imageX * imageY];

	//allocate memory for image matrix,
	//by adding 3 more levels
	imageM = new int[(imageX+2*apron) * (imageY + 2 * apron)];

	if (!readData) return 0;    // exit if failed allocation

	// open raw image file
	if (!loadRawImage16(fileName, imageX, imageY, readData))
	{
		delete[] readData;
		return 0;           // exit if failed to open image
	}

	int* vectorVM;
	vectorVM = new int[sizeOfVectorArray];
	int* filterV;
	filterV = new int[sizeOfVectorArray];

	high_resolution_clock::time_point t1 = high_resolution_clock::now();

	vectorVM = checkNeighbour(imageM,vectorVM, imageX + 2 * apron, imageY + 2 * apron);

	// filterV = filterStar(vectorVM, filterV, imageX + 2 * apron, imageY + 2 * apron);

	high_resolution_clock::time_point t2 = high_resolution_clock::now();
	auto duration = duration_cast<milliseconds>(t2 - t1).count();
	std::cout << duration << " miliseconds \n";

	FILE *outfile;

	// open file for writing
	outfile = fopen("lena16bit.bin", "w");
	if (outfile == NULL)
	{
		fprintf(stderr, "\nError opend file\n");
		exit(1);
	}
	for (int i = 0; i < sizeOfVectorArray; i++) {
		//std::cout << (filterV[i]);
		// fwrite(&(imageM[i]), sizeof(int), 1, outfile);
		fwrite(&(vectorVM[i]), 2, 1, outfile);


	}

	//if (fwrite != 0)
	//	printf("contents to file written successfully !\n");
	//else
	//	printf("error writing file !\n");

	free(imageM);
	free(vectorVM);
	free(filterV);
	return 0;
} // END OF MAIN //////////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////////////////////////
// load 16-bit greyscale RAW image
///////////////////////////////////////////////////////////////////////////////
bool loadRawImage16(const char *fileName, int x, int y, unsigned short *data)
{
	// check params
	if (!fileName || !data)
		return false;

	FILE *fp;
	if ((fp = fopen(fileName, "r")) == NULL)
	{
		printf("Cannot open %s.\n", fileName);
		return false;
	}

	// read pixel data
	fread(data, 2, x*y * 2, fp);
	fclose(fp);

	//// swap byte order
	// unsigned char byte;
	// for (int i = 0; i < x*y * 2; i += 2)
	// {
	// 	byte = data[i];
	// 	data[i] = data[i + 1];
	// 	data[i + 1] = byte;
	// }
	//change endian order from litte to big
	int rowN = -1;
	int rowNN = 0;
	int iN = 0;
	int width = y + 2 * apron;
	//first apron rows as 0
	for (int i = 0; i < apron*width; i++) {
		imageM[i] = 0;
	}
	for (int i = 0; i < x*y; i++) {
		rowNN = i / y;
		iN = i % y;
		if (rowNN != rowN) {
			for (int j = 0; j < apron; j++)
				imageM[(rowNN + apron)*width - j- 1] = 0;
			// new row started
			for (int j = 0; j < apron;j++)
				imageM[(rowNN + apron)*width + j] = 0;
			rowN++;
		}
		//unsigned short temp = _byteswap_ushort(data[i]);
		unsigned short temp = data[i];
		//assume that only apron pixels are zero
		//divide it by 100 for better visualization
		imageM[(rowNN+apron)*width+apron+iN] = temp+1; //extend the data to integer type
	}

	for (int i = (rowNN+apron)*width+width-apron; i < width*width; i++) {
		imageM[i] = 0;
	}
	//for (int i = 0; i < width*width; i++) {
	//	if (imageM[i] < 0) {
	//		int a = 0;
	//	}
	//}
	return true;
	free(data);
}

//check neighbour brightness level and set the tag of the pixel
int* checkNeighbour(int* matrix, int* vectorV, int x, int y) {
	// x is the image width plus 2 times the apron
	// first find the indices in the lower triangle

	// iterate through all the pixels
	// or through only the lower traingle
	int row, column;

	int matrixA [5][5];
	int matrixB [5][5];
	int product [5][5];
	int i,j,k,l;

	for (i = 0; i <x*y; i++) {

		row = i / x;
		column = i % x;

		if (matrix[i] == 0) {
			vectorV[i] = -1;
			continue;
		}
		else {
			vectorV[i] = matrix[i];
		}
		
		//check if it is in the lower triangle
		if(column < row){
			//initialize them
			for(l=0; l<5; ++l)
			for(j=0; j<5; ++j) {
				product[l][j] = 0;
				matrixA[l][j] = 0;
				matrixB[l][j] = 0;
			}
			// get matrixA
			getFivetoFiveMatrix(vectorV, matrixA,row,column,x);
			//get matrix B
			getFivetoFiveMatrix(vectorV, matrixB,column,row,x);

			
			for(l=0; l<5; ++l)
			for(j=0; j<5; ++j)
			for(k=0; k<5; ++k) {
				product[l][j]+=matrixA[l][k]*matrixB[k][j];

			}
			// add product into matrixA position of vectorV
			addMatrix(vectorV, product,row,column,x);
			
		}
		// vectorV[i] = findMaxNeigbour(matrix, i,y);
	

	}

	return vectorV;
}
// considering that the matrices are quadratic
int (*(getFivetoFiveMatrix)(int* vector, int matrix[][5], int row, int column, int rowLength))[5]
{
	int i,j;
	for(i=0;i<5;i++){
		for (j=0;j<5;j++){
			matrix[i][j]=vector[(row-2+i)*rowLength+(column-2+j)];
		}	
	}
	return matrix;
}

int (*(addMatrix)(int* vector, int matrix[][5], int row, int column, int rowLength))[5]
{
	int i,j;
	for(i=0;i<5;i++){
		for (j=0;j<5;j++){
			vector[(row-2+i)*rowLength+(column-2+j)]+=matrix[i][j];
		}
	
	}
	return matrix;
}

//
//void freeMemory(int* array, int size) {
//	if (array) {
//		for (int i = 0; i<size; i++) {
//				free(array[i]);
//		}
//		free(array);
//	}
//}











