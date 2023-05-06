#include<stdio.h>
#include<stdlib.h>



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


void setup_inputs(float **U, float **a, int **n, int *N, char *FILENAME)
{
	FILE *FP = fopen(FILENAME, "r");
	
	int lines_count = count_lines(FILENAME);

	char space;
	int U_len = 4;
	int a_len = *N;
	
	//read quantum gate
	fscanf(FP, "%f %f", *U, *U+1);
	fscanf(FP, "%f %f", *U+2, *U+3);

	//read blank
	fscanf(FP, "%c", space);

	// read a
	for(int i=0; i<a_len; i++)
		fscanf(FP,"%f", (*a+i));

	//read blank
	fscanf(FP, "%c", space);

	// read n
	fscanf(FP, "%d", *n);
	++(*(*n));

      fclose(FP);

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

		
		// destination index mapped via global thread id
		int dest_index_1 = src_index_1;
		int dest_index_2 = src_index_2;

		// result computation based on thread id mapped indices
		result[dest_index_1] = (a[src_index_1] * U[0]) + (a[src_index_2] * U[1]);
		result[dest_index_2] = (a[src_index_1] * U[2]) + (a[src_index_2] * U[3]);
	}	

	
	
}

int main(int argc, char *argv[])
{
	if(argc != 2)
	{
		printf("\n arguments mismatch. exiting");
	}

	char *IP_FILENAME = argv[1];
	
	// host as well as device copies of data
	int *n;
	int *N;
	float *a, *U, *result;


	//timestamps
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);


	// calculate and allocate space for N and n
	int size_n = sizeof(int);
	int size_N = sizeof(int);
	
	cudaMallocManaged((void **)&n,size_n);
	cudaMallocManaged((void **)&N, size_N);

	int lines_count = count_lines(IP_FILENAME);
	*N = lines_count - 5;

	// allocate space for U,a and result

	int size_oprnds = (*N) * sizeof(float);
	int size_U = 4 * sizeof(float);
	cudaMallocManaged((void **)&a,size_oprnds);
	cudaMallocManaged((void **)&result, size_oprnds);
	cudaMallocManaged((void **)&U, size_U);


	// setup inputs
	setup_inputs( &U, &a, &n, N, IP_FILENAME);
	


	//################### Device code #################################

	
    // launch kernel
	dim3 threadsPerBlock(256,1);
	
	dim3 numBlocks((*N)/threadsPerBlock.x + 1,1);
	qubit_gate<<<numBlocks, threadsPerBlock>>>(n, a, U, result, N);
	
	cudaEventRecord(stop);

	cudaEventSynchronize(stop);
	// // //#################################################################

	float execute_time = 0;
	cudaEventElapsedTime(&execute_time, start, stop);
	
	for(int i=0; i< (*N); i++)
           printf("%.3f\n", result[i]);
    //printf("\n Execution time : %f", execute_time);

	// cleanup
	cudaFree(n);
	cudaFree(a);
	cudaFree(U);
	cudaFree(result);
	cudaFree(N);
	
	return 0;

	
}