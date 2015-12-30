#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <omp.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "params.cuh"

double dRand(double, double);
inline int offset(int, int, int);
int parseCmdLineArgs(int, char**);
void cudaFiniteDiff(double*, double*);
void computePhysics(double*);
int queryDevices();

struct cudaDeviceProp deviceProp;

__device__ int dOffset(int x, int y, int z) {
	return z + (Z * y) + (Z * Y * x);
}
__device__ int sOffset(int x, int y, int z) {
	return z + ((BLOCKSIZE_Z + 2) * y) + ((BLOCKSIZE_Z + 2) * (BLOCKSIZE_Y + 4) * x);
}

__global__ void finiteDiff(double *dData) {

	int gi = blockIdx.x * blockDim.x + threadIdx.x; //global thread index - X
	int gj = blockIdx.y * blockDim.y + threadIdx.y; //global thread index - Y
	int gk = blockIdx.z * blockDim.z + threadIdx.z; //global thread index - Z

	int li = threadIdx.x; //local thread index - X
	int lj = threadIdx.y; //local thread index - Y
	int lk = threadIdx.z; //local thread index - Z

	// shared data + space for halo elements
	/* shared memory usage with 2x2x16 blocksize: (2+4)*(2+4)*(16+2)*8 = 5184 bytes. */
	__shared__ double sData[(BLOCKSIZE_X + 4)*(BLOCKSIZE_Y + 4)*(BLOCKSIZE_Z + 2)];

	// load a [BLOCKSIZE_X][BLOCKSIZE_Y][Z] block of data from global memory to shared.
	sData[sOffset(li+2,lj+2,lk+1)] = dData[dOffset(gi,gj,gk)];

	// copy left and right halo elements from global memory
	if (blockIdx.x == 0) {
		// copy left periodic halo elements
		if (li < 2) {
			sData[sOffset(li,lj+2,lk+1)] = dData[dOffset(gi+(X-2),gj,gk)];
		}
	}

	if (blockIdx.x > 0) {
		// copy left halo elements
		if (li < 2) {
			sData[sOffset(li,lj+2,lk+1)] = dData[dOffset(gi-2,gj,gk)];
		}
	}

	if (blockIdx.x == (gridDim.x - 1)) {
		//copy right periodic halo elements
		if (li >= BLOCKSIZE_X - 2) {
			sData[sOffset(li+4,lj+2,lk+1)] = dData[dOffset(gi-(X-2),gj,gk)];
		}
	}

	if (blockIdx.x < (gridDim.x - 1)) {
		//copy right halo elements
		if (li >= BLOCKSIZE_X - 2) {
			sData[sOffset(li+4,lj+2,lk+1)] = dData[dOffset(gi+2,gj,gk)];
		}
	}

	// copy top and bottom halo elements from global memory
	if (blockIdx.y == 0) {
		// copy top periodic halo elements
		if (lj < 2) {
			sData[sOffset(li+2,lj,lk+1)] = dData[dOffset(gi,gj+(Y-2),gk)];
		}
	}

	if (blockIdx.y > 0) {
		// copy top halo elements
		if (lj < 2) {
			sData[sOffset(li+2,lj,lk+1)] = dData[dOffset(gi,gj-2,gk)];
		}
	}

	if (blockIdx.y == (gridDim.y - 1)) {
		// copy bottom periodic halo elements
		if (lj >= BLOCKSIZE_Y - 2) {
			sData[sOffset(li+2,lj+4,lk+1)] = dData[dOffset(gi,gj-(Y-2),gk)];
		}
	}

	if (blockIdx.y < (gridDim.y - 1)) {
		// copy bottom halo elements
		if (lj >= BLOCKSIZE_Y - 2) {
			sData[sOffset(li+2,lj+4,lk+1)] = dData[dOffset(gi,gj+2,gk)];
		}
	}

	__syncthreads(); // ensure all halo elements are available

	// compute finite difference
	dData[dOffset(gi,gj,gk)] = 
		(4 * sData[sOffset(li+2,lj+2,lk+1)] + sData[sOffset(li+2 - 1,lj+2,lk+1)] 
	+ sData[sOffset(li+2,lj+2 - 1,lk+1)] + sData[sOffset(li+2 - 2,lj+2,lk+1)] 
	+ sData[sOffset(li+2 + 1,lj+2,lk+1)] + sData[sOffset(li+2,lj+2 + 1,lk+1)] 
	+ sData[sOffset(li+2 + 2,lj+2,lk+1)] + sData[sOffset(li+2,lj+2 + 2,lk+1)] 
	+ sData[sOffset(li+2,lj+2 - 2,lk+1)] + sData[sOffset(li+2,lj+2,lk+1 + 1)] 
	+ sData[sOffset(li+2,lj+1,lk+1 - 1)]) / 14;
}

int steps;
float Dt;

int main(int argc, char** argv) {

	double *data, *results;

	if (parseCmdLineArgs(argc, argv) == 1) {
		return 1;
	}

	if (queryDevices() == 1) {
		printf("\nRunning with %dx%dx%d problem size.\n", X, Y, Z);
		printf("\nDoing %d iterations.\n", steps);
		data = (double*) calloc(X*Y*Z, sizeof(double));
		results = (double*) calloc(X*Y*Z, sizeof(double));
		checkMalloc(data);
		srand((unsigned int)time(NULL));
		for (int i = 0; i < X; i++) {
			for (int j = 0; j < Y; j++) {
				for (int k = 0; k < Z; k++) {
					data[offset(i, j, k)] = dRand(10, 1000);
				}
			}
		}

		cudaFiniteDiff(data, results);

		printf("\nElapsed time: Dt = %.3f msec.\n", Dt);
		printf("Average Bandwidth (GB/s): %.3f\n",
			2.f * 1e-6 * X * Y * Z * steps * sizeof(double) / Dt);
	} else {
		printf("\nNo CUDA enabled devices found!\n");
		return 1;
	}

	return 0;
}

double dRand(double dMin, double dMax) {
	double d = (double) rand() / RAND_MAX;
	return dMin + d * (dMax - dMin);
}

inline int offset(int x, int y, int z) {
	return z + (Z * y) + (Z * Y * x);
}

void cudaFiniteDiff(double *data, double *results) {

	int i = 0;
	double *dData;
	size_t size = X*Y*Z*sizeof(double);

	// allocate memory for the arrays on device
	checkCuda(cudaMalloc(&dData, size));
	// copy data to device
	checkCuda(cudaMemcpy(dData, data, size, cudaMemcpyHostToDevice));

	// create a timing event
	cudaEvent_t start, stop;

	checkCuda(cudaEventCreate(&start));
	checkCuda(cudaEventCreate(&stop));

	// default is 64 threads per block
	dim3 blockSize(BLOCKSIZE_X, BLOCKSIZE_Y, BLOCKSIZE_Z);
	int numBlocks_X = X / blockSize.x;
	int numBlocks_Y = Y / blockSize.y;
	int numBlocks_Z = Z / blockSize.z;
	dim3 gridSize(numBlocks_X, numBlocks_Y, numBlocks_Z);

	/* on devices of compute capability 3.0 or higher, set shared memory bank size to eight bytes.
	* This can reduce bank conflicts when accessing double precision data. */
	cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);

	// start the timing
	checkCuda(cudaEventRecord(start));
	/* spawn 2 threads: one handles the kernel launches while the other computes physics on CPU */
#pragma omp parallel num_threads(2)
	{
		if (omp_get_thread_num() == 0) {
			for (i = 0; i < steps; i++) { 
				// launch the kernel that computes the Jacobi finite difference on GPU
				finiteDiff<<<gridSize, blockSize>>>(dData);
			}
		} else if (omp_get_thread_num() == 1) {
			for (i = 0; i < steps; i++) { 
				computePhysics(results);
			}
		}
	}
	// stop timing
	checkCuda(cudaEventRecord(stop));

	// get elapsed time
	checkCuda(cudaEventSynchronize(stop));
	checkCuda(cudaEventElapsedTime(&Dt, start, stop));
	checkCuda(cudaEventDestroy(start));
	checkCuda(cudaEventDestroy(stop));

	// copy calculated arrays from device to host
	checkCuda(cudaMemcpy(data, dData, size, cudaMemcpyDeviceToHost));
	checkCuda(cudaFree(dData));
	cudaDeviceReset();
}

// compute physics function. Data dependency is restricted within vertical columns.
void computePhysics(double *results) {
#pragma omp parallel for
	for (int i = 0; i < X; i++) {
#pragma omp parallel for
		for (int j = 0; j < Y; j++) {
			results[offset(i, j, 0)] = pow(S, 10);
			for (int k = 1; k < Z; k++) {
				results[offset(i, j, k)] = pow(
					0.9 * results[offset(i, j, k - 1)], 10);
			}
		}
	}
}

// check available CUDA-capable devices on the system
int queryDevices() {
	int deviceCount;
	cudaGetDeviceCount(&deviceCount);
	if (deviceCount > 0) {
		int device = 0; // we only use 1 GPU
		checkCuda(cudaGetDeviceProperties(&deviceProp, device));
		printf(
			"\nGPU: %s with Compute Capability %d.%d, %d KB shared / %d MB global memory\n",
			deviceProp.name, deviceProp.major, deviceProp.minor,
			deviceProp.sharedMemPerBlock / 1024,
			deviceProp.totalGlobalMem / 1048576);
		return 1;
	} else {
		return 0;
	}
}

int parseCmdLineArgs(int argc, char** argv) {
	if(argv[1] != NULL && strcmp(argv[1], "-steps") == 0) {
		if (argv[2] != NULL) {
			steps = atoi(argv[2]);
		} else {
			printf("\nMust specify number of iterations.\n");
			return 1;
		}
	} else {
		printf("\nMust specify number of iterations.\n");
		return 1;
	}
	return 0;
}
