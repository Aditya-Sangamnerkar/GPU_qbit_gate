#include<stdio.h>
#include<stdlib.h>

#define no_of_qgates 6

int count_lines(char *FILENAME)
{	
	int count=0;
	char c;
	FILE *FP = fopen(FILENAME, "r");
	while(c != EOF)
	{
		c = getc(FP);
		if(c == '\n')
			count++;
	}
	count++;
	fclose(FP);
	return count;
}

void setup_inputs(float U[no_of_qgates][4], 
				  float **a, 
				  int n[no_of_qgates], 
				  int *N, 
				  char *FILENAME)
{
	FILE *FP = fopen(FILENAME, "r");

	int lines_count = count_lines(FILENAME);
	char space;

	int U_len = 4;
	int a_len = lines_count - ((2 + 1) * no_of_qgates) - (no_of_qgates) -1;	// 3 lines/qgate; 1 line/nthbit
	*N = a_len;

	*a = (float *)malloc(a_len*sizeof(float));

	// read six quantum gates
	for(int i=0 ;i<no_of_qgates; i++)
	{
		fscanf(FP, "%f %f", &U[i][0], &U[i][1]);
		fscanf(FP, "%f %f", &U[i][2], &U[i][3]);
		fscanf(FP, "%c", &space);		// read blank space
	}

	// for(int i=0; i<no_of_qgates; i++)
	// {
	// 	printf("\n%0.3f %0.3f", U[i][0], U[i][1]);
	// 	printf("\n%0.3f %0.3f", U[i][2], U[i][3]);
	// 	printf("\n");
	// }

	// read a
	for(int i=0; i<a_len; i++)
		fscanf(FP, "%f", *a+i);

	// for(int i=0; i<a_len; i++)
	// 	printf("\n%0.3f", a[i]);

	// read blank space
	fscanf(FP, "%c", &space);

	// read six n's
	for(int i=0; i<no_of_qgates; i++){
		fscanf(FP, "%d", &n[i]);
		n[i]++;
	}

	// for(int i=0; i<no_of_qgates; i++)
	// 	printf("\n%d", n[i]);	
}

__global__ void qubit_gate(int *n, float *a, float *U, float *result, int *N)
{
	// global thread id
	int global_tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	int src_index_1, src_index_2, two_pow_n, two_pow_n_minus_one;

	if(global_tid < *N/2)
	{	
		two_pow_n = 1 << *n;
		two_pow_n_minus_one = 1 << (*n-1);
		
		// source index mapped via global thread id
		src_index_1 = (global_tid % two_pow_n_minus_one) + (two_pow_n * (global_tid / two_pow_n_minus_one));
		src_index_2 = src_index_1 + two_pow_n_minus_one;

		//printf("\n threadid : %d, src_index_1 : %d, src_index_2 : %d", global_tid, src_index_1, src_index_2);
		// destination index mapped via global thread id
		int dest_index_1 = src_index_1;
		int dest_index_2 = src_index_2;

		// // result computation based on thread id mapped indices
		result[dest_index_1] = (a[src_index_1] * U[0]) + (a[src_index_2] * U[1]);
		result[dest_index_2] = (a[src_index_1] * U[2]) + (a[src_index_2] * U[3]);
	}

	// if(global_tid == 0)
	// {
	// 	printf("\n N : %d", *N);
	// 	printf("\n n : %d", *n);
	// 	printf("\n %0.3f %0.3f", U[0], U[1]);
	// 	printf("\n %0.3f %0.3f", U[2], U[3]);

	// }
	
}

int main(int argc, char *argv[])
{	
	if(argc != 2)
	{
		printf("\n arguments mismatch. exiting");
		exit(0);
	}

	char *IP_FILENAME = argv[1];

	// host copies

	int n[no_of_qgates];		// array of nth qubits on which qgate is applied.
	int N;						// size of input vector
	float *a;					// input vector
	float *result;				// output vector
	float U[no_of_qgates][4];	// array of qgates

	// setup input values
	// allocate spaces for host copies

	setup_inputs(U, &a, n, &N, IP_FILENAME);

	int size_n = no_of_qgates * sizeof(int); 
	int size_oprnds = N * sizeof(float);
	int size_U = 4 * sizeof(float); 
	int size_N = sizeof(int);

	result = (float *)malloc(size_oprnds);
	
	// device copies
	int *d_n, *d_N;
	float *d_a, *d_U_1, *d_U_2, *d_U_3, *d_U_4, *d_U_5, *d_U_6;
	float *d_result_1, *d_result_2, *d_result_3, *d_result_4, *d_result_5, *d_result;

	//################### Device code #################################

	// allocate spaces for device copies
	cudaMalloc((void **)&d_n,size_n);
	cudaMalloc((void **)&d_a,size_oprnds);
	cudaMalloc((void **)&d_result, size_oprnds);
	cudaMalloc((void **)&d_result_1, size_oprnds);
	cudaMalloc((void **)&d_result_2, size_oprnds);
	cudaMalloc((void **)&d_result_3, size_oprnds);
	cudaMalloc((void **)&d_result_4, size_oprnds);
	cudaMalloc((void **)&d_result_5, size_oprnds);
	cudaMalloc((void **)&d_U_1, size_U);
	cudaMalloc((void **)&d_U_2, size_U);
	cudaMalloc((void **)&d_U_3, size_U);
	cudaMalloc((void **)&d_U_4, size_U);
	cudaMalloc((void **)&d_U_5, size_U);
	cudaMalloc((void **)&d_U_6, size_U);
	cudaMalloc((void **)&d_N, size_N);

	// copy inputs to device from host
	cudaMemcpy(d_n, n, size_n, cudaMemcpyHostToDevice);
	cudaMemcpy(d_a, a, size_oprnds, cudaMemcpyHostToDevice);
	cudaMemcpy(d_U_1, U[0], size_U, cudaMemcpyHostToDevice);
	cudaMemcpy(d_U_2, U[1], size_U, cudaMemcpyHostToDevice);
	cudaMemcpy(d_U_3, U[2], size_U, cudaMemcpyHostToDevice);
	cudaMemcpy(d_U_4, U[3], size_U, cudaMemcpyHostToDevice);
	cudaMemcpy(d_U_5, U[4], size_U, cudaMemcpyHostToDevice);
	cudaMemcpy(d_U_6, U[5], size_U, cudaMemcpyHostToDevice);
	cudaMemcpy(d_N, &N, size_N, cudaMemcpyHostToDevice);

	// launch kernel
	dim3 threadsPerBlock(1024,1);
	dim3 numBlocks(N/threadsPerBlock.x + 1,1);
	qubit_gate<<<numBlocks, threadsPerBlock>>>(d_n, d_a, d_U_1, d_result_1, d_N);
	cudaDeviceSynchronize();
	qubit_gate<<<numBlocks, threadsPerBlock>>>(d_n+1, d_result_1, d_U_2, d_result_2, d_N);
	cudaDeviceSynchronize();
	qubit_gate<<<numBlocks, threadsPerBlock>>>(d_n+2, d_result_2, d_U_3, d_result_3, d_N);
	cudaDeviceSynchronize();
	qubit_gate<<<numBlocks, threadsPerBlock>>>(d_n+3, d_result_3, d_U_4, d_result_4, d_N);
	cudaDeviceSynchronize();
	qubit_gate<<<numBlocks, threadsPerBlock>>>(d_n+4, d_result_4, d_U_5, d_result_5, d_N);
	cudaDeviceSynchronize();
	qubit_gate<<<numBlocks, threadsPerBlock>>>(d_n+5, d_result_5, d_U_6, d_result, d_N);

	// copy output to host from device
	cudaMemcpy(result, d_result, size_oprnds, cudaMemcpyDeviceToHost);

	for(int i=0; i<N; i++)
        printf("%.3f\n", result[i]);

    // cleanup
	cudaFree(d_n);
	cudaFree(d_a);
	cudaFree(d_U_1);
	cudaFree(d_U_2);
	cudaFree(d_U_3);
	cudaFree(d_U_4);
	cudaFree(d_U_5);
	cudaFree(d_U_6);
	cudaFree(d_result_1);
	cudaFree(d_result_2);
	cudaFree(d_result_3);
	cudaFree(d_result_4);
	cudaFree(d_result_5);
	cudaFree(d_result);
	free(result);
	free(a);
	return 0;
}