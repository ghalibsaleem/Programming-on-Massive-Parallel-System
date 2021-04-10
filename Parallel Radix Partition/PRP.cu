#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>


#define RAND_RANGE(N) ((double)rand()/((double)RAND_MAX + 1)*(N))

//data generator

void dataGenerator(int* data, int count, int first, int step)
{
	assert(data != NULL);

	for(int i = 0; i < count; ++i)
		data[i] = first + i * step;
	srand(time(NULL));
	for(int i = count-1; i>0; i--) //knuth shuffle
	{
        int j = RAND_RANGE(i);
        int k_tmp = data[i];
        data[i] = data[j];
        data[j] = k_tmp;
	}
}


/* This function embeds PTX code of CUDA to extract bit field from x. 
   "start" is the starting bit position relative to the LSB. 
   "nbits" is the bit field length.
   It returns the extracted bit field as an unsigned integer.
*/

__device__ uint bfe(uint x, uint start, uint nbits)
{
	uint bits;
	asm("bfe.u32 %0, %1, %2, %3;" : "=r"(bits) : "r"(x), "r"(start), "r"(nbits));
	return bits;
}


//define the histogram kernel here
__global__ void histogram(int * inData, int rSize, int numPartition, int *outData )
{
	int stride = blockDim.x * gridDim.x;
	int block = blockIdx.x * blockDim.x + threadIdx.x;
	for (int i = block; i < rSize; i+=stride){
		int index = bfe(inData[i], 0, numPartition);
		atomicAdd(&outData[index], 1);
	}
}

__global__ void prefixScan(int number_part, int *data, int *output_prefix)
{
	extern __shared__ int sum[]; 	
	int thread = (blockIdx.x * blockDim.x) + threadIdx.x;
	if(thread < number_part){
		if(thread > 0)
			sum[thread] = data[thread-1];
		sum[0] = 0;
		__syncthreads();
		
		int offset = 1;
        while(offset< number_part){
			int bi = thread;
			int ai = thread - offset;
			if(ai <= 0)
				sum[bi] = sum[bi];
			else
				sum[bi] += sum[ai];
			
			__syncthreads();
			offset *= 2;
		}
		output_prefix[thread] = sum[thread];
	}
}

__global__ void Reorder(int *input_data, int *ouput_prefix, int rSize, int number_part, int *output_reorder) {
	int block = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = block; i < rSize; i+=stride){
		i = input_data[i];
		int index = bfe(i, 0, number_part);
		int offset = atomicAdd((int*)ouput_prefix[index], 1);
		atomicAdd(&output_reorder[offset], 1);
	}
}

//Method added to report the running time of the kernal
struct timezone Idunno;	
struct timeval startTime, endTime;
double report_running_time_GPU() {
	long sec_diff, usec_diff;
	gettimeofday(&endTime, &Idunno);
	sec_diff = endTime.tv_sec - startTime.tv_sec;
	usec_diff= endTime.tv_usec-startTime.tv_usec;
	if(usec_diff < 0) {
		sec_diff --;
		usec_diff += 1000000;
	}
	
	printf("Running Time for all kernals: %ld.%06lds\n", sec_diff, usec_diff);
	return (double)(sec_diff*1.0 + usec_diff/1000000.0);
}

//Method to check cuda errors
void cuda_eror_check(cudaError_t errors, const char out[]){
	if (errors != cudaSuccess)
    {
        printf("There is something wrong with cuda %s, %s, \n", out, cudaGetErrorString(errors));
        exit(EXIT_FAILURE);
    }
}

int main(int argc, char const *argv[]){

	// Condition Check for arguments before initilizing anything
	if (argc != 3 ){
		printf("Incorrect Input!\n input should be of form: ./outputFileName {#of_elements_in_array} {#of_partitions}\n");
		exit(0);
	}
	
	// Declaring all the required Variables
	int blocksize = 64;
	int rSize = atoi(argv[1]);
	int number_part = atoi(argv[2]);
	int count = (int)ceil( (double)rSize/ (double) blocksize);
	int bits =(int)log2((double)number_part);
	int *r_h, *histogram_data, *prefix, *reorder;

	//Cuda error check while allocating r_h
	cuda_eror_check(cudaMallocHost((void**)&r_h, sizeof(int)*rSize), "Use Pinned Memory" ); 
	
	//data generator
	dataGenerator(r_h, rSize, 0, 1);

	//cuda error check when allocating histogram, reorder, prefix
	cuda_eror_check(cudaMallocHost((void **) &histogram_data, sizeof(int)* number_part), "Cuda Malloc for Histogram");
	cuda_eror_check(cudaMallocHost((void **) &reorder, sizeof(int)* rSize), "Cuda Malloc for Reorder");
	cuda_eror_check(cudaMallocHost((void **) &prefix, sizeof(int)* number_part), "Cuda Malloc for Prefix");
	
	//Start the time to calculate running time
	gettimeofday(&startTime, &Idunno);
	
	
	histogram<<<count, blocksize>>>(r_h, rSize, bits, histogram_data);
	cuda_eror_check(cudaDeviceSynchronize(), "histogram function" );

	
	prefixScan<<<1, number_part, sizeof(int)*number_part>>>(number_part, histogram_data, prefix);
	cuda_eror_check(cudaDeviceSynchronize(), "prefixScan function" );

	
	Reorder<<<count, blocksize>>>(r_h,prefix , rSize, number_part, reorder);
	
	//Loop for printing the output
	for(int i = 0; i < number_part; i++){
		printf("Partition %d:  ", i);
		printf("Offset: %d  ", prefix[i]);
		printf("Number of Keys: %d\n", histogram_data[i]);
	}
	printf("\n");

	//Report Running time
	report_running_time_GPU();

	//Free all the variable
	cudaFreeHost(r_h);
	cudaFreeHost(histogram_data);
	cudaFreeHost(prefix);
	cudaFreeHost(reorder);
	
	return 0;
}
