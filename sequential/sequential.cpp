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
int findMaxNeigbour(int* matrix, int index, int width);
int* filterStar(int* matrix, int* output, int x, int y);
int calculateVectorSum(int* input, int width, int index);
Vector2d vectorDir(int x);
int closestDir(Vector2d vec);
float Rad2Deg(float radians);
//void freeMemory(int* array, int size);

// CONSTANTS //////////////////////////////////////////////////////////////////
const char *FILE_NAME = "/home/ziya/parallel-task/lena16bit.raw";
const int MAX_NAME = 512;               // max length of string
const int IMAGE_X = 256;                // image width
const int IMAGE_Y = 256;                // image height
const int apron = 3;
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

	filterV = filterStar(vectorVM, filterV, imageX + 2 * apron, imageY + 2 * apron);

	high_resolution_clock::time_point t2 = high_resolution_clock::now();
	auto duration = duration_cast<milliseconds>(t2 - t1).count();
	std::cout << duration << " miliseconds \n";

	FILE *outfile;

	// open file for writing
	outfile = fopen("ducks.bin", "w");
	if (outfile == NULL)
	{
		fprintf(stderr, "\nError opend file\n");
		exit(1);
	}
	for (int i = 0; i < sizeOfVectorArray; i++) {
		//std::cout << (filterV[i]);
		fwrite(&(filterV[i]), sizeof(int), 1, outfile);

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
	fread(data, 1, x*y * 2, fp);
	fclose(fp);

	//// swap byte order
	//unsigned char byte;
	//for (int i = 0; i < x*y * 2; i += 2)
	//{
	//	byte = data[i];
	//	data[i] = data[i + 1];
	//	data[i + 1] = byte;
	//}
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
	//what if putting 0's at the edges

	//int rowN;
	//int iN;
	//int index;
	for (int i = 0; i <x*y; i++) {
	/*	rowN = i / y;
		iN = i % y;
		index = (y + 2 * apron)*(rowN + apron) + (iN + apron);*/

		//only apron values are zero, already have added to every 16 bit value "one"
		if (matrix[i] == 0) {
			vectorV[i] = -1;
			continue;
		}
		else {
			vectorV[i] = findMaxNeigbour(matrix, i,y);
		}

	}

	return vectorV;
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

int findMaxNeigbour(int* matrix, int index,int width) {
	int max = matrix[index];
	int tag = 0;

	int n = matrix[index - width];
	int ne = matrix[index - width + 1];
	int e = matrix[index + 1];
	int se = matrix[index + width + 1];
	int s = matrix[index + width];
	int sw = matrix[index + width - 1];
	int w = matrix[index - 1];
	int nw = matrix[index - width - 1];

	if (n > max) {
		max = n;
		tag=1;
	}
	if (ne > max) {
		max = ne;
		tag=2;
	}
	if (e > max) {
		max = e;
		tag=3;
	}
	if (se > max) {
		max = se;
		tag=4;
	}
	if (s > max) {
		max = s;
		tag=5;
	}
	if (sw > max) {
		max = n;
		tag=6;
	}
	if (w > max) {
		max = n;
		tag=7;
	}
	if (nw > max) {
		max = n;
		tag=8;
	}
	return tag;
}

int* filterStar(int* matrix,int* output,int x,int y) {

	for (int i = 0; i <x*y; i++) {

		//only apron values are zero, already have added to every 16 bit value "one"
		if (matrix[i] == -1) {
			output[i] = -1;
			continue;
		}
		else {
			output[i] = calculateVectorSum(matrix, y, i);
		}

	}

	return output;
}

int calculateVectorSum(int* input, int width, int index) {
	int dir = 0;
	Vector2d sum = Vector2d(0, 0);
	sum = sum + Vector2d(1, 1);
	for (int i = 1; i < 4; i++) {
		sum = sum + vectorDir(input[index - (width*i)] ) + vectorDir(input[index + (width*i)] );
		sum = sum + vectorDir(input[index-i]) + vectorDir(input[index+i]);
		for (int j = 1; j < 5-i; j++) {
			sum = sum + vectorDir(input[index - width*i + j]) + vectorDir(input[index - width*i - j]);
		}
	}
	
	float length = Vector2d::Magnitude(sum);	
	if (length < THRESHHOLD)
		return 0;	
	sum = Vector2d::Normal(sum);
	return closestDir(sum);
}

int closestDir(Vector2d vec) {
	float angle = atan2(vec.x, vec.y);
	angle = Rad2Deg(angle);
	if (angle < 0) angle = 360 + angle;
	if (angle < 22.5 || angle >337.5) return 1;
	else if (angle < 67.5) return 2;
	else if (angle < 112.5) return 3;
	else if (angle < 157.5) return 4;
	else if (angle < 202.5) return 5;
	else if (angle < 247.5) return 6;
	else if (angle < 292.5) return 7;
	else return 8;
}

float Rad2Deg(float radians) {
	return radians * (180 / 3.141592653589793238);
}

Vector2d vectorDir(int x) {
	switch (x) {
	case 0:
		return Vector2d(0, 0);
	case 1:
		return Vector2d(0, 1);
	case 2:
		return Vector2d(0.707, 0.707);
	case 3:
		return Vector2d(1, 0);
	case 4:
		return Vector2d(0.707,-0.707);
	case 5:
		return Vector2d(0,-1);
	case 6:
		return Vector2d(-0.707,-0.707);
	case 7:
		return Vector2d(-1,0);
	case 8:
		return Vector2d(-0.707, 0.707);
	case -1:
		return Vector2d(0, 0);
	//default:
	//	return Vector2d(0, 0);
	}
}









