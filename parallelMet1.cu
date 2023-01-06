#include <string.h>
#include <x86intrin.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include <cuda_runtime.h>
// #include <device_functions.h>

// static void HandleError(cudaError_t err,
// 	const char *file,
// 	int line) {
// 	if (err != cudaSuccess) {
// 		printf("%s in %s at line %d\n", cudaGetErrorString(err),
// 			file, line);
// 		exit(EXIT_FAILURE);
// 	}
// }
// #define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))



bool loadRawImage16(const char *, int, int, unsigned short *, int*);
cudaError_t allocateMemoryOnDevice(int *, int *);
__global__ void checkNeigbours(int*, int*, int, int);


//Constants
const char *FILE_NAME = "/home/ziya/parallel-task/lena16bit.raw";
const int MAX_NAME = 512;               // max length of string
const int IMAGE_X = 256;                // image width
const int IMAGE_Y = 256;                // image height
const int APRON = 2;
const int WIDTH = (IMAGE_X + APRON * 2);
const int SIZEOFVECTORARRAY = (IMAGE_X + APRON * 2) * (IMAGE_X + APRON * 2);
//const float THRESHHOLD = 3;

__constant__ int THRESHOLD = 3;

__global__ void checkNeigbours(int* dataMatrix, int* vectorMatrix, int X, int Y) {
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = j * X + i;

	int row, column;

	int matrixA [5][5];
	int matrixB [5][5];
	int product [5][5];
	int k,l,h;

	if ((i < X) && (j < Y)) {

		row = index / X;
		column = index % X;

		if (dataMatrix[index] == 0) {
			vectorMatrix[index] = -1;
		}
		else {
			vectorMatrix[index] = dataMatrix[index];
		}

		//check if it is in the lower triangle
		if(column < row){
			//initialize them
			for(l=0; l<5; ++l)
			for(h=0; h<5; ++h) {
				product[l][h] = 0;
				matrixA[l][h] = 0;
				matrixB[l][h] = 0;
			}
			// get matrixA
			// getFivetoFiveMatrix(vectorV, matrixA,row,column,x);
			for(l=0;l<5;l++){
				for (h=0;h<5;h++){
					matrixA[l][h]=dataMatrix[(row-2+l)*X+(column-2+h)];
				}	
			}
			//get matrix B
			// getFivetoFiveMatrix(vectorV, matrixB,column,row,x);
			for(l=0;l<5;l++){
				for (h=0;h<5;h++){
					matrixB[l][h]=dataMatrix[(row-2+l)*X+(column-2+h)];
				}	
			}
			
			for(l=0; l<5; ++l)
			for(h=0; h<5; ++h)
			for(k=0; k<5; ++k) {
				product[l][h]+=matrixA[l][k]*matrixB[k][h];

			}
			// add product into matrixA position of vectorV
			// addMatrix(vectorV, product,row,column,x);
			for(l=0;l<5;l++){
				for (h=0;h<5;h++){
					vectorMatrix[(row-2+l)*X+(column-2+h)]+=product[l][h];
				}
	
			}
			
		}

	}
	//__syncthreads();
	if (i<2 * APRON || i>X - 2 * APRON) return;
	if (j<2 * APRON || j>Y - 2 * APRON) return;
	return;
}


int main()
{
	cudaError_t cudaStatus;
	char fileName[MAX_NAME];
	unsigned short* readData;
	// allocate memory for image data, 16 bits per pixel
	readData = new unsigned short[IMAGE_X * IMAGE_Y];

	// image matrix
	int* imageM;
	//allocate memory for image matrix,
	//by adding 3 more levels
	imageM = new int[SIZEOFVECTORARRAY];

	strcpy(fileName, FILE_NAME);
	// open raw image file
	if (!loadRawImage16(fileName, IMAGE_X, IMAGE_Y, readData, imageM))
	{
		delete[] readData;
		return 0;           // exit if failed to open image
	}

	//the vector image
	int* imageV;
	imageV = new int[SIZEOFVECTORARRAY];

	//time metrics
	float elapsed = 0;
	cudaEvent_t start, stop;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start, 0);

	cudaStatus = allocateMemoryOnDevice(imageM, imageV);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "allocateMemoryOnDevice failed!");
		return 1;
	}
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&elapsed, start, stop);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);


	printf("The elapsed time in gpu was %.10f ms\", elapsed");

	//write into a file
	FILE *outfile;

	// open file for writing
	outfile = fopen("ducks-16.bin", "w");
	if (outfile == NULL)
	{
		fprintf(stderr, "\nError opend file\n");
		exit(1);
	}
	for (int i = 0; i < SIZEOFVECTORARRAY; i++) {
		//std::cout << (filterV[i]);
		// fwrite(&(imageV[i]), sizeof(int), 1, outfile);
		fwrite(&(imageV[i]), 2, 1, outfile);

	}

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}
	free(imageM);
	free(imageV);

	return 0;
}
cudaError_t allocateMemoryOnDevice(int * matrixA, int * matrixV) {
	cudaError_t cudaStatus;

	int* d_matrixA;
	int* d_matrixV;

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&d_matrixA, SIZEOFVECTORARRAY * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		// goto Error;
	}
	cudaStatus = cudaMalloc((void**)&d_matrixV, SIZEOFVECTORARRAY * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		// goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(d_matrixA, matrixA, SIZEOFVECTORARRAY * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		// goto Error;
	}

	dim3 threadsPerBlock(16,16);
	dim3 numBlocks(WIDTH / threadsPerBlock.x,
		WIDTH / threadsPerBlock.y);

	// int threadsPerBlock = 128;
	// int numBlocks;
	// numBlocks =(68644 + threadsPerBlock - 1) / threadsPerBlock;


	// Launch a kernel on the GPU with one thread for each element.
	checkNeigbours << <numBlocks, threadsPerBlock >> >(d_matrixA, d_matrixV, IMAGE_X + APRON * 2, IMAGE_Y + APRON * 2);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "checkNeigbours launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching checkNeigbours!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(matrixV, d_matrixV, SIZEOFVECTORARRAY * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(d_matrixA);
	cudaFree(d_matrixV);

	return cudaStatus;
}

///////////////////////////////////////////////////////////////////////////////
// load 16-bit greyscale RAW image
///////////////////////////////////////////////////////////////////////////////
bool loadRawImage16(const char *fileName, int x, int y, unsigned short *data, int* newMat)
{
	// check params
	if (!fileName || !data)
		return false;

	FILE *fp;
	if ((fp = fopen(fileName, "r")) == NULL)
	{
		printf("Cannot open %s.\n", fileName);
		// perror( "Failed to open hello.txt" );
		return false;
	}

	// read pixel data
	fread(data, 1, x*y * 2, fp);
	fclose(fp);

	//change endian order from litte to big
	int rowN = -1;
	int rowNN = 0;
	int iN = 0;
	int width = y + 2 * APRON;
	//first apron rows as 0
	for (int i = 0; i < APRON*width; i++) {
		newMat[i] = 0;
	}
	for (int i = 0; i < x*y; i++) {
		rowNN = i / y;
		iN = i % y;
		if (rowNN != rowN) {
			for (int j = 0; j < APRON; j++)
				newMat[(rowNN + APRON)*width - j - 1] = 0;
			// new row started
			for (int j = 0; j < APRON; j++)
				newMat[(rowNN + APRON)*width + j] = 0;
			rowN++;
		}
		//unsigned short temp = _byteswap_ushort(data[i]);
		unsigned short temp = data[i];
		//assume that only apron pixels are zero
		newMat[(rowNN + APRON)*width + APRON + iN] = temp + 1; //extend the data to integer type
	}

	for (int i = (rowNN + APRON)*width + width - APRON; i < width*width; i++) {
		newMat[i] = 0;
	}

	free(data);
	return true;

}

