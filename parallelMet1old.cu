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
__device__ int findMaxNeighbour(int*, int, int);
__device__ int findVectorSum(int*, int, int);
__device__ float* vectorDir(int, float [2]);
__device__ int closestDir(float*);


//Constants
const char *FILE_NAME = "/home/ziya/parallel-task/lena16bit.raw";
const int MAX_NAME = 512;               // max length of string
const int IMAGE_X = 256;                // image width
const int IMAGE_Y = 256;                // image height
const int APRON = 3;
const int WIDTH = (IMAGE_X + APRON * 2);
const int SIZEOFVECTORARRAY = (IMAGE_X + APRON * 2) * (IMAGE_X + APRON * 2);
//const float THRESHHOLD = 3;

__constant__ int THRESHOLD = 3;

__global__ void checkNeigbours(int* dataMatrix, int* vectorMatrix, int X, int Y) {
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = j * X + i;
	if ((i < X) && (j < Y)) {
		vectorMatrix[index] = findMaxNeighbour(dataMatrix, index, X);
	}
	//__syncthreads();
	if (i<2 * APRON || i>X - 2 * APRON) return;
	if (j<2 * APRON || j>Y - 2 * APRON) return;

	vectorMatrix[index] = findVectorSum(vectorMatrix, index, X);


}

__device__ int findVectorSum(int* matrix, int index, int X) {
	float sum[2] = { 0 };
	float vec[2];
	for (int i = 1; i < 4; i++) {
		float * vec1 = vectorDir(matrix[index - (X*i)], vec);
		float * vec2 = vectorDir(matrix[index + (X*i)], vec);
		sum[0] = sum[0] + vec1[0] + vec2[0];
		sum[1] = sum[1] + vec1[1] + vec2[1];
		vec1 = vectorDir(matrix[index - i], vec);
		vec2 = vectorDir(matrix[index + i], vec);
		sum[0] = sum[0] + vec1[0] + vec2[0];
		sum[1] = sum[1] + vec1[1] + vec2[1];
		for (int j = 1; j < 5 - i; j++) {
			vec1 = vectorDir(matrix[index - X * i + j], vec);
			vec2 = vectorDir(matrix[index - X * i - j], vec);
			sum[0] = sum[0] + vec1[0] + vec2[0];
			sum[1] = sum[1] + vec1[1] + vec2[1];
		}
	}
	float length = sqrtf((sum[0] * sum[0]) + (sum[1] * sum[1]));
	if (length < THRESHOLD)
		return 0;
	return closestDir(sum);
}

__device__ int closestDir(float* vec) {
	float angle = atan2(vec[0], vec[1]);
	angle = angle * (180 / 3.141592653589793238);
	if (angle < 0) angle = 360 + angle;
	angle = angle + 45;
	int dir = angle / 45;
	if ((int)angle % 45 > 22.5) dir = dir + 1;
	if (dir == 9) return 0;
	else return dir;
}


__device__ float* vectorDir(int X,float vec[]) {
	vec[0] = 0;
	vec[1] = 1;
	if (X == 0) {
		return vec;
		// return {0,0}
	}else if (X == 1){
		vec[0] = 0; vec[1] = 1;
		return vec;}
		else if (X == 2){
		vec[0] = 0.707; vec[1] = 0.707;
		return vec;}
		else if (X == 3){
		vec[0] = 1; vec[1] = 0;
		return vec;}
		else if (X == 4){
		vec[0] = 0.707; vec[1] = -0.707;
		return vec;}
		else if (X == 5){
		vec[0] = 0; vec[1] = -1;
		return vec;}
		else if (X == 6){
		vec[0] = -0.707; vec[1] = -0.707;
		return vec;}
		else if (X == 7){
		vec[0] = -1; vec[1] = 0;
		return vec;}
		else if (X == 8){
		vec[0] = -0.707; vec[1] = 0.707;
		return vec;
		}
		else if (X == 1){
		vec[0] = 0; vec[1] = 0;
		return vec;}
	else {return vec;}
}
__device__ int findMaxNeighbour(int* matrix, int index, int X) {
	int max = matrix[index];
	if (max == 0) return -1;
	int tag = 0;

	int n = matrix[index - X];
	int ne = matrix[index - X + 1];
	int e = matrix[index + 1];
	int se = matrix[index + X + 1];
	int s = matrix[index + X];
	int sw = matrix[index + X - 1];
	int w = matrix[index - 1];
	int nw = matrix[index - X - 1];

	if (n > max) {
		max = n;
		tag = 1;
	}
	if (ne > max) {
		max = ne;
		tag = 2;
	}
	if (e > max) {
		max = e;
		tag = 3;
	}
	if (se > max) {
		max = se;
		tag = 4;
	}
	if (s > max) {
		max = s;
		tag = 5;
	}
	if (sw > max) {
		max = n;
		tag = 6;
	}
	if (w > max) {
		max = n;
		tag = 7;
	}
	if (nw > max) {
		max = n;
		tag = 8;
	}
	return tag;
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
		fwrite(&(imageV[i]), sizeof(int), 1, outfile);

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