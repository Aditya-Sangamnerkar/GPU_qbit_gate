#include<stdio.h>
#include<stdlib.h>
#include<math.h>

#define no_of_qgates 6 // modify


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


void setup_inputs(float **U, float **a, int **n, int *N, int **inactive_n, char *FILENAME)
{
	FILE *FP = fopen(FILENAME, "r");
	
	int lines_count = count_lines(FILENAME);
	char space;
	
	int U_len = 4 * no_of_qgates;
	int a_len = lines_count - ((2 + 1) * no_of_qgates) - (no_of_qgates) -1;
	*N = a_len;

	*U = (float *)malloc(U_len*sizeof(float));
	*a = (float *)malloc(a_len*sizeof(float));
	*n = (int *)malloc(no_of_qgates*sizeof(int));

	for(int i=0; i<4*no_of_qgates; i=i+4)
	{
		//read quantum gate
		fscanf(FP, "%f %f", *U+i, *U+i+1);
		fscanf(FP, "%f %f", *U+i+2, *U+i+3);

		//read blank
		fscanf(FP, "%c", &space);
	}
	

	// read a
	for(int i=0; i<a_len; i++)
		fscanf(FP,"%f", (*a+i));

	//read blank
	fscanf(FP, "%c", &space);

	// read n
	for(int i=0; i<no_of_qgates; i++){
		fscanf(FP, "%d", (*n+i));
	}

    fclose(FP);

}


__global__ void qubit_gate(int *n, float *a, float *U, float *result, int *N, int *inactive_n, int *inactive_N)
{	
	// shared memory
	__shared__ float a_shared[64];  // modify

	// ***************** LOAD DATA INTO SHARED MEMORY ***********************

	// inactive mask
	int inactive_mask = 0;

	for(int i=0; i<*inactive_N; i++)
	{
		// extract bit
		int bit_i = (blockIdx.x >> i) & 1;
		// position it at inactive index
		inactive_mask = inactive_mask | (bit_i << inactive_n[i]);
	}

	// active mask one
	int active_mask_one = 0;
	int shared_index_one = 2 * threadIdx.x;

	for(int i=0; i<no_of_qgates; i++)
	{
		//extract bit
		int bit_i = (shared_index_one >> i) & 1;
		// position it at active index
		active_mask_one = active_mask_one | (bit_i << n[i]); 
	}

	int global_index_one = active_mask_one | inactive_mask;

	int active_mask_two = 0;
	int shared_index_two = shared_index_one + 1;

	for(int i=0; i<no_of_qgates; i++)
	{
		//extract bit
		int bit_i = (shared_index_two >> i) & 1;
		// position it at active index
		active_mask_two = active_mask_two | (bit_i << n[i]); 
	}

	int global_index_two = active_mask_two | inactive_mask;

	

	a_shared[shared_index_one] = a[global_index_one];
	a_shared[shared_index_two] = a[global_index_two];

	__syncthreads();

	//printf("\n bId : %d tId : %d g_index_1 : %d g_index_2 : %d", blockIdx.x, threadIdx.x, global_index_one, global_index_two);
	
	// **********************************************

	int shared_src_index_one, shared_src_index_two;
	float result_one, result_two;
	int two_pow_t;
	int two_pow_t_plus_one;
	int t;
	int u_index;

	for(int i=0; i<no_of_qgates; i++)
    {
        t = i;
        two_pow_t = 1 << t;
        two_pow_t_plus_one = two_pow_t * 2;
        u_index = i * 4;
        shared_src_index_one = (threadIdx.x % two_pow_t) + (two_pow_t_plus_one * (threadIdx.x / two_pow_t));
        shared_src_index_two =  shared_src_index_one + two_pow_t;

        result_one = (a_shared[shared_src_index_one] * U[u_index]) + (a_shared[shared_src_index_two] * U[u_index+1]);
        result_two = (a_shared[shared_src_index_one] * U[u_index+2]) + (a_shared[shared_src_index_two] * U[u_index+3]);
        __syncthreads();
        a_shared[shared_src_index_one] = result_one;
        a_shared[shared_src_index_two] = result_two;
        __syncthreads();
        //printf("\n bId : %d tId : %d t : %d sh_index_1 : %d sh_index_2 : %d val1 : %0.3f val2 : %0.3f", blockIdx.x, threadIdx.x, t, shared_src_index_one, shared_src_index_two, result_one, result_two);    

    }

    // ***************** LOAD DATA INTO GLOBAL MEMORY ***********************
	result[global_index_one] = a_shared[shared_index_one];
	result[global_index_two] = a_shared[shared_index_two];	

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
	int *n;
	int N, inactive_N;
	float *a, *U, *result;
	int *inactive_n;


	// setup input values
	// allocate spaces for host copies 

	setup_inputs( &U, &a, &n, &N, &inactive_n, IP_FILENAME);

	inactive_N = (log(N)/log(2)) - no_of_qgates;
	inactive_n = (int *)malloc(inactive_N*sizeof(int));

	int i=0;
	int qgate_index = 0;
	bool flag = true;
	while(i < inactive_N)
	{
		for(int j=0; j< no_of_qgates; j++)		
		{
			if(qgate_index == n[j])
			{
				qgate_index++;
				flag = false;
			}
		}

		if(flag)
		{
			inactive_n[i] = qgate_index;
			i++;
			qgate_index++;
		}
		flag = true;
	}

	int size_inactive_n = inactive_N * sizeof(int);
	int size_n = no_of_qgates * sizeof(int);
	int size_oprnds = N * sizeof(float);
	int size_U = 4 * no_of_qgates *sizeof(float);
	int size_N = sizeof(int);
	int size_inactive_N = sizeof(int);

	result = (float *)malloc(size_oprnds);	

	// device copies
	int *d_n, *d_N, *d_inactive_n, *d_inactive_N;
	float *d_a, *d_U, *d_result;
	
	//################### Device code #################################

	// allocate spaces for device copies 
	cudaMalloc((void **)&d_n,size_n);
	cudaMalloc((void **)&d_a,size_oprnds);
	cudaMalloc((void **)&d_result, size_oprnds);
	cudaMalloc((void **)&d_U, size_U);
	cudaMalloc((void **)&d_N, size_N);
	cudaMalloc((void **)&d_inactive_n,size_inactive_n);
	cudaMalloc((void **)&d_inactive_N,size_inactive_N);
	
	// copy inputs to device from host
	cudaMemcpy(d_n, n, size_n, cudaMemcpyHostToDevice);
	cudaMemcpy(d_a, a, size_oprnds, cudaMemcpyHostToDevice);
	cudaMemcpy(d_U, U, size_U, cudaMemcpyHostToDevice);
	cudaMemcpy(d_N, &N, size_N, cudaMemcpyHostToDevice);
	cudaMemcpy(d_inactive_N, &inactive_N, size_inactive_N, cudaMemcpyHostToDevice);
	cudaMemcpy(d_inactive_n, inactive_n, size_inactive_n, cudaMemcpyHostToDevice);
	
	

	// launch kernel
	dim3 threadsPerBlock(32,1);	// modify
	int blocks = N/64;			// 64 = 2^6 6 qubits always
	dim3 numBlocks(blocks,1);		// modify
	qubit_gate<<<numBlocks, threadsPerBlock>>>(d_n, d_a, d_U, d_result, d_N, d_inactive_n, d_inactive_N);

	// copy output to host from device
	cudaMemcpy(result, d_result, size_oprnds, cudaMemcpyDeviceToHost);
	
	//#################################################################


	for(int i=0; i<N; i++)
        printf("%.3f\n", result[i]);

    // cleanup
	cudaFree(d_n);
	cudaFree(d_a);
	cudaFree(d_U);
	cudaFree(d_result);
	free(result);
	free(a);
	free(U);
	free(n);
	free(inactive_n);


}