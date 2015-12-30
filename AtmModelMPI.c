#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "mpi.h"
#include "omp.h"

#define MASS 1073741824 // 2^30
#define X 1 // initial value of X, representing solar radiation in the vertical dimension
#define OMP 0 // defines whether OpenMP will be used, default is no.

int Nx, Ny, Nz; /* problem grid */
int width, height, depth; /* part of grid allocated to each process */
int p; /* number of processes */
int i; /* width */
int j; /* depth */
int k; /* height */
int totalSteps; /* user-specified number of iterations */
int reduceFreq; /* default is to perform reduction at every iteration */

int parseCmdLineArgs(int, char**, int*, int);
int offset(int, int, int);
double dRand(double dMin, double dMax);
void computeInnerJacobi(double*);
void computeOuterJacobi(double*);
void computeInnerPhysics(double*);
void computeOuterPhysics(double*);

int main(int argc, char* argv[]) {
	int myRank; /* rank of process in MPI_COMM_WORLD */
	int myGridRank; /* rank of process in the grid */
	int source, dest;
	int tag = 0; /* tag for messages */

	double totalMass; /* total mass across all processes */
	double massPerProcess;
	/* mass for each process */

	double t1 = 0, t2 = 0, Dt_local = 0, Dt_total = 0;
	double t1_comm_sendrecv = 0, t2_comm_sendrecv = 0, t1_comm_wait_recv = 0,
			t2_comm_wait_recv = 0, t1_comm_wait_send = 0, t2_comm_wait_send = 0,
			Dt_local_comm = 0, Dt_total_comm = 0;
	double t1_inner_comp = 0, t2_inner_comp = 0, t1_outer_comp = 0,
			t2_outer_comp = 0, Dt_local_comp = 0, Dt_total_comp = 0;
	int steps = 0; /* computation steps */

	/* Parameters for the 2-D cartesian grid */

	/* 2 dimensions */
	int ndims = 2;
	/* number of processes in each dimension */
	int dims[2];
	/* determines whether a dimension has cyclic boundaries,
	 * meaning the 2 edge processes are connected  */
	int periods[2];
	/* cartesian topology communicator */
	MPI_Comm comm_cart;
	/* process coordinates in the grid */
	int coords[2];
	/* coords of next neighboring process in the stencil */
	int nextCoords[2];

	/* part of the dataset assigned to this process */
	double *data = NULL;
	/* physics calcs results for each point will be stored here */
	double *results = NULL;

	/* array of identifiers for non-blocking recv and send */
	MPI_Request recvRequestArr[4], sendRequestArr[4];
	/* count how many non-blocking recvs and sends have been posted */
	int recvRequestCount = 0, sendRequestCount = 0;

	MPI_Errhandler errHandler;
	int errCode;

	/* start up MPI */
	MPI_Init(&argc, &argv);
	/* get number of processes */
	MPI_Comm_size(MPI_COMM_WORLD, &p);
	/* get process rank */
	MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

	if (parseCmdLineArgs(argc, argv, dims, myRank) == 1) {
		MPI_Finalize();
		return 1;
	}

	/* [width, depth, height] of the part of the problem's grid assigned to each
	 * process, calculated by the total number of nodes in each dimension divided by
	 * the number of processes assigned to that dimension. */
	width = (int) ceil((double) (Nx / dims[0])); // x
	depth = (int) ceil((double) (Ny / dims[1])); // y
	height = Nz; // z

	massPerProcess = MASS / p;

	if (myRank == 0) {
		printf("\nProblem grid: %dx%dx%d.\n", Nx, Ny, Nz);
		printf("P = %d processes.\n", p);
		printf("Sub-grid / process: %dx%dx%d.\n", width, depth, height);
		printf("\nC = %d iterations.\n", totalSteps);
		printf("Reduction performed every %d iteration(s).\n", reduceFreq);
	}

	/* There's communication wraparound in the atmosphere model, in the X and Y axis,
	 * so X and Y will have cyclic boundaries. */
	periods[0] = 1;
	periods[1] = 1;

	/* Create a 3-D cartesian communication grid and allow process rank
	 * reordering */
	MPI_Cart_create(MPI_COMM_WORLD, ndims, dims, periods, 1, &comm_cart);

	/* find out process rank in the grid */
	MPI_Comm_rank(comm_cart, &myGridRank);
	/* find out process coordinates in the grid */
	MPI_Cart_coords(comm_cart, myGridRank, ndims, coords);

	/* initialize dataset for this process plus added padding for halo points.
	 * Each node in the process creates its own random values.
	 * Also allocate space for the results from physics calculations performed at
	 * each point in the vertical direction */
	srand(myGridRank);
	data = calloc((width + 4) * (depth + 4) * (height + 2), sizeof(double));
	results = calloc((width + 4) * (depth + 4) * (height + 2), sizeof(double));
	for (i = 2; i < width + 2; i++) {
		for (j = 2; j < depth + 2; j++) {
			for (k = 1; k < height + 1; k++) {
				/* generate random value from 10 to 1000 for node (i,j,k) */
				data[offset(i, j, k)] = dRand(10, 1000);
			}
		}
	}

	/* create datatypes, by defining sections of the original 3-D array.
	 * These sections will be sent and received to/from the corresponding neighbors */

	int array_of_sizes[3];
	int array_of_subsizes[3];
	int array_of_starts[3];
	int sub_dims = 3;

	/* slice to be sent to left neighbor (X-axis) */
	MPI_Datatype SEND_TO_LEFT;
	array_of_sizes[0] = width + 4;
	array_of_sizes[1] = depth + 4;
	array_of_sizes[2] = height + 2;
	array_of_subsizes[0] = 2;
	array_of_subsizes[1] = depth;
	array_of_subsizes[2] = height;
	array_of_starts[0] = 2;
	array_of_starts[1] = 2;
	array_of_starts[2] = 1;
	MPI_Type_create_subarray(sub_dims, array_of_sizes, array_of_subsizes,
			array_of_starts, MPI_ORDER_C, MPI_DOUBLE, &SEND_TO_LEFT);
	MPI_Type_commit(&SEND_TO_LEFT);

	/* slice to be received from left neighbor (X-axis) */
	MPI_Datatype RCV_FROM_LEFT;
	array_of_sizes[0] = width + 4;
	array_of_sizes[1] = depth + 4;
	array_of_sizes[2] = height + 2;
	array_of_subsizes[0] = 2;
	array_of_subsizes[1] = depth;
	array_of_subsizes[2] = height;
	array_of_starts[0] = 0;
	array_of_starts[1] = 2;
	array_of_starts[2] = 1;
	MPI_Type_create_subarray(sub_dims, array_of_sizes, array_of_subsizes,
			array_of_starts, MPI_ORDER_C, MPI_DOUBLE, &RCV_FROM_LEFT);
	MPI_Type_commit(&RCV_FROM_LEFT);

	/* slice to be sent to back neighbor (Y-axis) */
	MPI_Datatype SEND_TO_BACK;
	array_of_sizes[0] = width + 4;
	array_of_sizes[1] = depth + 4;
	array_of_sizes[2] = height + 2;
	array_of_subsizes[0] = width;
	array_of_subsizes[1] = 2;
	array_of_subsizes[2] = height;
	array_of_starts[0] = 2;
	array_of_starts[1] = 2;
	array_of_starts[2] = 1;
	MPI_Type_create_subarray(sub_dims, array_of_sizes, array_of_subsizes,
			array_of_starts, MPI_ORDER_C, MPI_DOUBLE, &SEND_TO_BACK);
	MPI_Type_commit(&SEND_TO_BACK);

	/* slice to be received from back neighbor (Y-axis) */
	MPI_Datatype RCV_FROM_BACK;
	array_of_sizes[0] = width + 4;
	array_of_sizes[1] = depth + 4;
	array_of_sizes[2] = height + 2;
	array_of_subsizes[0] = width;
	array_of_subsizes[1] = 2;
	array_of_subsizes[2] = height;
	array_of_starts[0] = 2;
	array_of_starts[1] = 0;
	array_of_starts[2] = 1;
	MPI_Type_create_subarray(sub_dims, array_of_sizes, array_of_subsizes,
			array_of_starts, MPI_ORDER_C, MPI_DOUBLE, &RCV_FROM_BACK);
	MPI_Type_commit(&RCV_FROM_BACK);

	/* slice to be sent to front neighbor (Y-axis) */
	MPI_Datatype SEND_TO_FRONT;
	array_of_sizes[0] = width + 4;
	array_of_sizes[1] = depth + 4;
	array_of_sizes[2] = height + 2;
	array_of_subsizes[0] = width;
	array_of_subsizes[1] = 2;
	array_of_subsizes[2] = height;
	array_of_starts[0] = 2;
	array_of_starts[1] = depth;
	array_of_starts[2] = 1;
	MPI_Type_create_subarray(sub_dims, array_of_sizes, array_of_subsizes,
			array_of_starts, MPI_ORDER_C, MPI_DOUBLE, &SEND_TO_FRONT);
	MPI_Type_commit(&SEND_TO_FRONT);

	/* slice to be received from front neighbor (Y-axis) */
	MPI_Datatype RCV_FROM_FRONT;
	array_of_sizes[0] = width + 4;
	array_of_sizes[1] = depth + 4;
	array_of_sizes[2] = height + 2;
	array_of_subsizes[0] = width;
	array_of_subsizes[1] = 2;
	array_of_subsizes[2] = height;
	array_of_starts[0] = 2;
	array_of_starts[1] = depth + 2;
	array_of_starts[2] = 1;
	MPI_Type_create_subarray(sub_dims, array_of_sizes, array_of_subsizes,
			array_of_starts, MPI_ORDER_C, MPI_DOUBLE, &RCV_FROM_FRONT);
	MPI_Type_commit(&RCV_FROM_FRONT);

	/* slice to be sent to right neighbor (X-axis) */
	MPI_Datatype SEND_TO_RIGHT;
	array_of_sizes[0] = width + 4;
	array_of_sizes[1] = depth + 4;
	array_of_sizes[2] = height + 2;
	array_of_subsizes[0] = 2;
	array_of_subsizes[1] = depth;
	array_of_subsizes[2] = height;
	array_of_starts[0] = width;
	array_of_starts[1] = 2;
	array_of_starts[2] = 1;
	MPI_Type_create_subarray(sub_dims, array_of_sizes, array_of_subsizes,
			array_of_starts, MPI_ORDER_C, MPI_DOUBLE, &SEND_TO_RIGHT);
	MPI_Type_commit(&SEND_TO_RIGHT);

	/* slice to be received from right neighbor (X-axis) */
	MPI_Datatype RCV_FROM_RIGHT;
	array_of_sizes[0] = width + 4;
	array_of_sizes[1] = depth + 4;
	array_of_sizes[2] = height + 2;
	array_of_subsizes[0] = 2;
	array_of_subsizes[1] = depth;
	array_of_subsizes[2] = height;
	array_of_starts[0] = width + 2;
	array_of_starts[1] = 2;
	array_of_starts[2] = 1;
	MPI_Type_create_subarray(sub_dims, array_of_sizes, array_of_subsizes,
			array_of_starts, MPI_ORDER_C, MPI_DOUBLE, &RCV_FROM_RIGHT);
	MPI_Type_commit(&RCV_FROM_RIGHT);

	/* The predefined default error handler, which is
	 * MPI_ERRORS_ARE_FATAL, for a newly created communicator or
	 * for MPI_COMM_WORLD is to abort the  whole parallel program
	 * as soon as any MPI error is detected. By setting the error handler to
	 * MPI_ERRORS_RETURN  the program will no longer abort on having detected
	 * an MPI error, instead the error will be returned so we can handle it. */
	MPI_Errhandler_get(comm_cart, &errHandler);
	if (errHandler == MPI_ERRORS_ARE_FATAL) {
		MPI_Errhandler_set(comm_cart, MPI_ERRORS_RETURN);
	}

	/* locate neighboring processes in the grid and initiate a send-receive
	 * operation for each neighbor found */

	/* process (x-1,y) */
	nextCoords[0] = coords[0] - 1;
	nextCoords[1] = coords[1];
	errCode = MPI_Cart_rank(comm_cart, nextCoords, &dest);
	if (errCode == MPI_SUCCESS) {
		MPI_Send_init(&data[offset(0, 0, 0)], 1, SEND_TO_LEFT, dest, tag,
				comm_cart, &sendRequestArr[sendRequestCount]);
		sendRequestCount++;
		source = dest;
		MPI_Recv_init(&data[offset(0, 0, 0)], 1, RCV_FROM_LEFT, source, tag,
				comm_cart, &recvRequestArr[recvRequestCount]);
		recvRequestCount++;
	}
	/* process (x+1,y) */
	nextCoords[0] = coords[0] + 1;
	nextCoords[1] = coords[1];
	errCode = MPI_Cart_rank(comm_cart, nextCoords, &dest);
	if (errCode == MPI_SUCCESS) {
		MPI_Send_init(&data[offset(0, 0, 0)], 1, SEND_TO_RIGHT, dest, tag,
				comm_cart, &sendRequestArr[sendRequestCount]);
		sendRequestCount++;
		source = dest;
		MPI_Recv_init(&data[offset(0, 0, 0)], 1, RCV_FROM_RIGHT, source, tag,
				comm_cart, &recvRequestArr[recvRequestCount]);
		recvRequestCount++;
	}
	/* process (x,y+1) */
	nextCoords[0] = coords[0];
	nextCoords[1] = coords[1] + 1;
	errCode = MPI_Cart_rank(comm_cart, nextCoords, &dest);
	if (errCode == MPI_SUCCESS) {
		MPI_Send_init(&data[offset(0, 0, 0)], 1, SEND_TO_BACK, dest, tag,
				comm_cart, &sendRequestArr[sendRequestCount]);
		sendRequestCount++;
		source = dest;
		MPI_Recv_init(&data[offset(0, 0, 0)], 1, RCV_FROM_BACK, source, tag,
				comm_cart, &recvRequestArr[recvRequestCount]);
		recvRequestCount++;
	}
	/* process (x,y-1) */
	nextCoords[0] = coords[0];
	nextCoords[1] = coords[1] - 1;
	errCode = MPI_Cart_rank(comm_cart, nextCoords, &dest);
	if (errCode == MPI_SUCCESS) {
		MPI_Send_init(&data[offset(0, 0, 0)], 1, SEND_TO_FRONT, dest, tag,
				comm_cart, &sendRequestArr[sendRequestCount]);
		sendRequestCount++;
		source = dest;
		MPI_Recv_init(&data[offset(0, 0, 0)], 1, RCV_FROM_FRONT, source, tag,
				comm_cart, &recvRequestArr[recvRequestCount]);
		recvRequestCount++;
	}

	t1 = MPI_Wtime();

	/* Compute the Jacobi finite difference, in parallel, in the horizontal dimension.
	 * For each step of the computation, the process sends its data to its neighbors
	 * and receives their own. Physics calculations are performed in the vertical
	 * dimension. With vertical agglomeration communication for those is avoided.
	 * Each iteration advances the computation by one time step. */
	while (steps < totalSteps) {
		steps++;

		t1_comm_sendrecv += MPI_Wtime();
		MPI_Startall(sendRequestCount, sendRequestArr);
		MPI_Startall(recvRequestCount, recvRequestArr);
		t2_comm_sendrecv += MPI_Wtime();

		t1_inner_comp += MPI_Wtime();
		/* compute inner, neighbor-independent, part while communication completes */
		computeInnerJacobi(data);
		/* do physics computations in the vertical dimension, for the inner part.
		 * Physics computations is the serial part of the algorithm. */
		computeInnerPhysics(results);
		t2_inner_comp += MPI_Wtime();

		t1_comm_wait_recv += MPI_Wtime();
		/* ensure all data have been successfully received from neighbors
		 * before computing outer points */
		MPI_Waitall(recvRequestCount, recvRequestArr, MPI_STATUSES_IGNORE);
		t2_comm_wait_recv += MPI_Wtime();

		t1_outer_comp += MPI_Wtime();
		/* compute Jacobi outer part */
		computeOuterJacobi(data);
		/* compute physics for the outer part */
		computeOuterPhysics(results);
		t2_outer_comp += MPI_Wtime();

		/* perform reduction with a given frequency */
		if (steps % reduceFreq == 0) {
			MPI_Reduce(&massPerProcess, &totalMass, 1, MPI_DOUBLE,
			MPI_SUM, 0, comm_cart);
			totalMass = totalMass / p;
		}

		t1_comm_wait_send += MPI_Wtime();
		/* ensure all data have been sent successfully sent
		 * before the next loop iteration */
		MPI_Waitall(sendRequestCount, sendRequestArr, MPI_STATUSES_IGNORE);
		t2_comm_wait_send += MPI_Wtime();

	}

	t2 = MPI_Wtime();

	/* find time taken for the process slowest to compute, the process slowest to
	 * communicate and the slowest process overall. */
	Dt_local = t2 - t1;
	MPI_Reduce(&Dt_local, &Dt_total, 1, MPI_DOUBLE, MPI_MAX, 0, comm_cart);
	Dt_local_comp = (t2_inner_comp - t1_inner_comp)
			+ (t2_outer_comp - t1_outer_comp);
	MPI_Reduce(&Dt_local_comp, &Dt_total_comp, 1, MPI_DOUBLE, MPI_MAX, 0,
			comm_cart);
	Dt_local_comm = (t2_comm_sendrecv - t1_comm_sendrecv)
			+ (t2_comm_wait_recv - t1_comm_wait_recv)
			+ (t2_comm_wait_send - t1_comm_wait_send);
	MPI_Reduce(&Dt_local_comm, &Dt_total_comm, 1, MPI_DOUBLE, MPI_MAX, 0,
			comm_cart);

	/* Elapsed times for the slowest process */
	if (myGridRank == 0) {
		printf("\nMax. Computation time: Dt = %.3f msec.\n",
				Dt_total_comp * 1000);
		printf("Max. Communication time: Dt = %.3f msec.\n",
				Dt_total_comm * 1000);
		printf("Max. Total time: Dt = %.3f msec.\n\n", Dt_total * 1000);
	}

	/* shut down MPI */
	MPI_Finalize();

	return 0;
}

/* utility function that returns the offset of an element in a 3D array allocated
 * in row major order */
int offset(int x, int y, int z) {
	return z + ((height + 2) * y) + ((height + 2) * (depth + 4) * x);
}

void computeInnerJacobi(double *data) {
#if OMP
#pragma omp parallel for collapse(3)
#endif
	/* compute finite difference for inner part */
	for (i = 4; i < width; i++) {
		for (j = 4; j < depth; j++) {
			for (k = 1; k < height + 1; k++) {
				data[offset(i, j, k)] = (4 * data[offset(i, j, k)]
						+ data[offset(i - 1, j, k)] + data[offset(i, j - 1, k)]
						+ data[offset(i - 2, j, k)] + data[offset(i, j - 2, k)]
						+ data[offset(i + 1, j, k)] + data[offset(i, j + 1, k)]
						+ data[offset(i + 2, j, k)] + data[offset(i, j + 2, k)]
						+ data[offset(i, j, k - 1)] + data[offset(i, j, k + 1)])
						/ 14;
			}
		}
	}
}

void computeOuterJacobi(double *data) {
#if OMP
#pragma omp parallel
#endif
	{
#if OMP
#pragma omp for collapse(3)
#endif
		/* left outer part */
		for (i = 2; i < 4; i++) {
			for (j = 2; j < depth + 2; j++) {
				for (k = 1; k < height + 1; k++) {
					data[offset(i, j, k)] = (4 * data[offset(i, j, k)]
							+ data[offset(i - 1, j, k)]
							+ data[offset(i, j - 1, k)]
							+ data[offset(i - 2, j, k)]
							+ data[offset(i, j - 2, k)]
							+ data[offset(i + 1, j, k)]
							+ data[offset(i, j + 1, k)]
							+ data[offset(i + 2, j, k)]
							+ data[offset(i, j + 2, k)]
							+ data[offset(i, j, k - 1)]
							+ data[offset(i, j, k + 1)]) / 14;
				}
			}
		}
#if OMP
#pragma omp for collapse(3)
#endif
		/* right outer part */
		for (i = width; i < width + 2; i++) {
			for (j = 2; j < depth + 2; j++) {
				for (k = 1; k < height + 1; k++) {
					data[offset(i, j, k)] = (4 * data[offset(i, j, k)]
							+ data[offset(i - 1, j, k)]
							+ data[offset(i, j - 1, k)]
							+ data[offset(i - 2, j, k)]
							+ data[offset(i, j - 2, k)]
							+ data[offset(i + 1, j, k)]
							+ data[offset(i, j + 1, k)]
							+ data[offset(i + 2, j, k)]
							+ data[offset(i, j + 2, k)]
							+ data[offset(i, j, k - 1)]
							+ data[offset(i, j, k + 1)]) / 14;
				}
			}
		}
#if OMP
#pragma omp for collapse(3)
#endif
		/* back outer part */
		for (i = 4; i < width; i++) {
			for (j = 2; j < 4; j++) {
				for (k = 1; k < height + 1; k++) {
					data[offset(i, j, k)] = (4 * data[offset(i, j, k)]
							+ data[offset(i - 1, j, k)]
							+ data[offset(i, j - 1, k)]
							+ data[offset(i - 2, j, k)]
							+ data[offset(i, j - 2, k)]
							+ data[offset(i + 1, j, k)]
							+ data[offset(i, j + 1, k)]
							+ data[offset(i + 2, j, k)]
							+ data[offset(i, j + 2, k)]
							+ data[offset(i, j, k - 1)]
							+ data[offset(i, j, k + 1)]) / 14;
				}
			}
		}
#if OMP
#pragma omp for collapse(3)
#endif
		/* front outer part */
		for (i = 4; i < width; i++) {
			for (j = depth; j < depth + 2; j++) {
				for (k = 1; k < height + 1; k++) {
					data[offset(i, j, k)] = (4 * data[offset(i, j, k)]
							+ data[offset(i - 1, j, k)]
							+ data[offset(i, j - 1, k)]
							+ data[offset(i - 2, j, k)]
							+ data[offset(i, j - 2, k)]
							+ data[offset(i + 1, j, k)]
							+ data[offset(i, j + 1, k)]
							+ data[offset(i + 2, j, k)]
							+ data[offset(i, j + 2, k)]
							+ data[offset(i, j, k - 1)]
							+ data[offset(i, j, k + 1)]) / 14;
				}
			}
		}
	}
}

void computeInnerPhysics(double *results) {
#if OMP
#pragma omp parallel for collapse(2)
#endif
	for (i = 4; i < width; i++) {
		for (j = 4; j < depth; j++) {
			/* top layer is hit by 100% of solar radiation */
			results[offset(i, j, 1)] = pow(X, 10);
			for (k = 2; k < height + 1; k++) {
				/* each subsequent layer absorbs 10% */
				results[offset(i, j, k)] = pow(
						0.9 * results[offset(i, j, k - 1)], 10);
			}
		}
	}
}

void computeOuterPhysics(double *results) {
#if OMP
#pragma omp parallel
#endif
	{
#if OMP
#pragma omp for collapse(2)
#endif
		/* left outer part */
		for (i = 2; i < 4; i++) {
			for (j = 2; j < depth + 2; j++) {
				results[offset(i, j, 1)] = pow(X, 10);
				for (k = 2; k < height + 1; k++) {
					results[offset(i, j, k)] = pow(
							0.9 * results[offset(i, j, k - 1)], 10);
				}
			}
		}
#if OMP
#pragma omp for collapse(2)
#endif
		/* right outer part */
		for (i = width; i < width + 2; i++) {
			for (j = 2; j < depth + 2; j++) {
				results[offset(i, j, 1)] = pow(X, 10);
				for (k = 2; k < height + 1; k++) {
					results[offset(i, j, k)] = pow(
							0.9 * results[offset(i, j, k - 1)], 10);
				}
			}
		}
#if OMP
#pragma omp for collapse(2)
#endif
		/* back outer part */
		for (i = 4; i < width; i++) {
			for (j = 2; j < 4; j++) {
				results[offset(i, j, 1)] = pow(X, 10);
				for (k = 2; k < height + 1; k++) {
					results[offset(i, j, k)] = pow(
							0.9 * results[offset(i, j, k - 1)], 10);
				}
			}
		}
#if OMP
#pragma omp for collapse(2)
#endif
		/* front outer part */
		for (i = 4; i < width; i++) {
			for (j = depth; j < depth + 2; j++) {
				results[offset(i, j, 1)] = pow(X, 10);
				for (k = 2; k < height + 1; k++) {
					results[offset(i, j, k)] = pow(
							0.9 * results[offset(i, j, k - 1)], 10);
				}
			}
		}
	}
}

/* generate random doubles between 2 values */
double dRand(double dMin, double dMax) {
	double d = (double) rand() / RAND_MAX;
	return dMin + d * (dMax - dMin);
}

int parseCmdLineArgs(int argc, char **argv, int *dims, int myRank) {
	if (argv[1] != NULL && strcmp(argv[1], "-nodes") == 0) {
		if (argv[2] != NULL && argv[3] != NULL && argv[4] != NULL) {
			Nx = atoi(argv[2]);
			Ny = atoi(argv[3]);
			Nz = atoi(argv[4]);
		} else {
			if (myRank == 0) {
				printf(
						"\nSpecify grid of nodes, grid of processes, number of iterations and reduction frequency"
								" [-nodes <Nx> <Ny> <Nz> -procs <i> <j> -steps <n> -reduce <f>]\n\n");
			}
			return 1;
		}
	} else {
		if (myRank == 0) {
			printf(
					"\nSpecify grid of nodes, grid of processes, number of iterations and reduction frequency"
							" [-nodes <Nx> <Ny> <Nz> -procs <i> <j> -steps <n> -reduce <f>]\n\n");
		}
		return 1;
	}

	/* allocate processes to each dimension. */
	if (argv[5] != NULL && strcmp(argv[5], "-procs") == 0) {
		if (argv[6] != NULL && argv[7] != NULL) {
			dims[0] = (int) atoi(argv[6]);
			dims[1] = (int) atoi(argv[7]);
		} else {
			if (myRank == 0) {
				printf(
						"\nSpecify grid of nodes, grid of processes, number of iterations and reduction frequency"
								" [-nodes <Nx> <Ny> <Nz> -procs <i> <j> -steps <n> -reduce <f>]\n\n");
			}
			return 1;
		}
	} else {
		if (myRank == 0) {
			printf(
					"\nSpecify grid of nodes, grid of processes, number of iterations and reduction frequency"
							" [-nodes <Nx> <Ny> <Nz> -procs <i> <j> -steps <n> -reduce <f>]\n\n");
		}
		return 1;
	}

	/* Grid of processes size must equal total number of processes */
	if (dims[0] * dims[1] != p) {
		if (myRank == 0) {
			printf("\nProcessing grid size must equal total number of processes"
					" (np = i*j).\n\n");
		}
		return 1;
	}

	/* specify number of iterations */
	if (argv[8] != NULL && strcmp(argv[8], "-steps") == 0) {
		if (argv[9] != NULL) {
			totalSteps = atoi(argv[9]);
		} else {
			if (myRank == 0) {
				printf(
						"\nSpecify grid of nodes, grid of processes, number of iterations and reduction frequency"
								" [-nodes <Nx> <Ny> <Nz> -procs <i> <j> -steps <n> -reduce <f>]\n\n");
			}
			return 1;
		}
	} else {
		if (myRank == 0) {
			printf(
					"\nSpecify grid of nodes, grid of processes, number of iterations and reduction frequency"
							" [-nodes <Nx> <Ny> <Nz> -procs <i> <j> -steps <n> -reduce <f>]\n\n");
		}
		return 1;
	}

	if (argv[10] != NULL && strcmp(argv[10], "-reduce") == 0) {
		if (argv[11] != NULL) {
			reduceFreq = (int) atoi(argv[11]);
		} else {
			if (myRank == 0) {
				printf(
						"\nSpecify grid of nodes, grid of processes, number of iterations and reduction frequency"
								" [-nodes <Nx> <Ny> <Nz> -procs <i> <j> -steps <n> -reduce <f>]\n\n");
			}
			return 1;
		}
	} else {
		if (myRank == 0) {
			printf(
					"\nSpecify grid of nodes, grid of processes, number of iterations and reduction frequency"
							" [-nodes <Nx> <Ny> <Nz> -procs <i> <j> -steps <n> -reduce <f>]\n\n");
		}
		return 1;
	}

	return 0;
}
