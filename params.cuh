#include <stdio.h>
#include <assert.h>
#include "cuda_runtime.h"

#ifndef PARAMS_H_
#define PARAMS_H_

inline
	void checkCuda(cudaError_t result)
{
	if (result != cudaSuccess) {
		fprintf(stderr, "\nCUDA Runtime Error: %s\n\n", cudaGetErrorString(result));
		assert(result == cudaSuccess);
	}
}

void checkMalloc(double *ptr) {
	if (ptr == NULL) {
		fprintf(stderr, "\nMemory allocation error.\n\n");
		assert(ptr != NULL);
	}
}

#define S 1

#define X 512
#define Y 512
#define Z 16

// default is 2*2*16 = 64 threads
#define BLOCKSIZE_X 2
#define BLOCKSIZE_Y 2
#define BLOCKSIZE_Z Z


#endif
