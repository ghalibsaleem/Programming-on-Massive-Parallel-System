#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

#include <cuda_runtime.h>
#include <cuda.h>


#define BOX_SIZE	23000 /* size of the data box on one dimension            */

/* descriptors for single atom in the tree */
typedef struct atomdesc {
	double x_pos;
	double y_pos;
	double z_pos;
} atom;

typedef struct hist_entry{
	//float min;
	//float max;
	unsigned long long d_cnt;   /* need a long long type as the count might be huge */
} bucket;


bucket * histogram;		/* list of all buckets in the histogram   */
long long	PDH_acnt;	/* total number of data points            */
int num_buckets;		/* total number of buckets in the histogram */
double   PDH_res;		/* value of w                             */
atom * atom_list;		/* list of all data points                */

/* These are for an old way of tracking time */
struct timezone Idunno;	
struct timeval startTime, endTime;


/* 
	distance of two points in the atom_list 
*/
double p2p_distance(int ind1, int ind2) {
	//TODO
	double x1 = atom_list[ind1].x_pos;
	double x2 = atom_list[ind2].x_pos;
	double y1 = atom_list[ind1].y_pos;
	double y2 = atom_list[ind2].y_pos;
	double z1 = atom_list[ind1].z_pos;
	double z2 = atom_list[ind2].z_pos;
		
	return sqrt((x1 - x2)*(x1-x2) + (y1 - y2)*(y1 - y2) + (z1 - z2)*(z1 - z2));
}

__device__ double p2p_distance(atom * atomList, int ind1, int ind2) {
	
	double x1 = atomList[ind1].x_pos;
	double x2 = atomList[ind2].x_pos;
	double y1 = atomList[ind1].y_pos;
	double y2 = atomList[ind2].y_pos;
	double z1 = atomList[ind1].z_pos;
	double z2 = atomList[ind2].z_pos;
		
	return sqrt((x1 - x2)*(x1-x2) + (y1 - y2)*(y1 - y2) + (z1 - z2)*(z1 - z2));
}


/* 
	brute-force SDH solution in a single CPU thread 
*/
int PDH_baseline() {
	//TODO
	int i, j, h_pos;
	double dist;
	
	for(i = 0; i < PDH_acnt; i++) {
		for(j = i+1; j < PDH_acnt; j++) {
			dist = p2p_distance(i,j);
			h_pos = (int) (dist / PDH_res);
			histogram[h_pos].d_cnt++;
		} 
	}
	return 0;
}



__global__ void PDH_baseline(bucket *histogram, atom * atomList, double width, int size) {
	int i, j, h_pos;
	double dist;

	i = blockIdx.x * blockDim.x + threadIdx.x;
	j = i + 1;

	for(int l_index = j; l_index < size; l_index++){
		dist = p2p_distance(atomList, i, l_index);
		h_pos = (int) (dist / width);
		atomicAdd(&histogram[h_pos].d_cnt, 1);
	}
	
	__syncthreads();
	/*
	for(i = 0; i < PDH_acnt; i++) {
		for(j = i+1; j < PDH_acnt; j++) {
			dist = p2p_distance(i,j);
			h_pos = (int) (dist / PDH_res);
			histogram[h_pos].d_cnt++;
		} 
	}
	return 0;
	*/
}


__global__ void PDH2D_basline_algorithm3(bucket *histogram, atom *atom_list, long long width, double size, int num_blocks, int num_buckets) {
	
	int block, thread, blockdim, blockid;
	atom left, right;
	
	extern __shared__ bucket sharedHistogram[];
	__shared__ atom sharedAtomList[256];

	blockid = blockIdx.x;	
	blockdim = blockDim.x;
	thread = threadIdx.x;
	block = blockIdx.x * blockDim.x + threadIdx.x;
	
	/*This will initialize the shared memory*/
	for(int i = thread; i < num_buckets; i += blockdim) {
		sharedHistogram[i].d_cnt = 0;
	}
	
	left = atom_list[block];
	__syncthreads();
	
	if(block < width) 
		for(int i = blockid + 1; i < num_blocks; i++) {		
			sharedAtomList[thread] = atom_list[i * blockdim + thread];
			__syncthreads();

			if(i*blockdim < width)
			for(int j = 0; j < blockdim; j++) {
				if(i*blockdim + j < width) {
					right = sharedAtomList[j];
					//double distance =sqrt(pow(left.x_pos - right.x_pos,2) + pow(left.y_pos - right.y_pos,2) + pow(left.z_pos - right.z_pos,2));
					double distance = sqrt((left.x_pos - right.x_pos)*(left.x_pos - right.x_pos) + (left.y_pos - right.y_pos)*(left.y_pos - right.y_pos) + (left.z_pos - right.z_pos)*(left.z_pos -right.z_pos));
					atomicAdd(&sharedHistogram[(int)(distance/size)].d_cnt,1);
				}
			}
			__syncthreads();
		}
	
	sharedAtomList[thread] = left;
	__syncthreads();

	if(block < width)
		for(int i = thread + 1; i < blockdim; i++) {
			
			if(blockdim * blockid + i < width) {
				right = sharedAtomList[i];
				//double distance =sqrt(pow(left.x_pos - right.x_pos,2) + pow(left.y_pos - right.y_pos,2) + pow(left.z_pos - right.z_pos,2));
				double distance = sqrt((left.x_pos - right.x_pos)*(left.x_pos - right.x_pos) + (left.y_pos - right.y_pos)*(left.y_pos - right.y_pos) + (left.z_pos - right.z_pos)*(left.z_pos - right.z_pos));
				atomicAdd(&sharedHistogram[(int)(distance/size)].d_cnt,1);
			}
		}
	__syncthreads();

	for(int i = thread; i < num_buckets; i += blockdim) {
		atomicAdd(&histogram[i].d_cnt,sharedHistogram[i].d_cnt);
	}
}


/* 
	set a checkpoint and show the (natural) running time in seconds 
*/
double report_running_time(int flag = 0) {
	long sec_diff, usec_diff;
	gettimeofday(&endTime, &Idunno);
	sec_diff = endTime.tv_sec - startTime.tv_sec;
	usec_diff= endTime.tv_usec-startTime.tv_usec;
	if(usec_diff < 0) {
		sec_diff --;
		usec_diff += 1000000;
	}
	if (flag == 0){
		printf("Running time for GPU Old version: %ld.%06ld\n", sec_diff, usec_diff);
	}
	else{
		printf("Running time for GPU new version: %ld.%06ld\n", sec_diff, usec_diff);
	}
	return (double)(sec_diff*1.0 + usec_diff/1000000.0);
}


/* 
	print the counts in all buckets of the histogram 
*/
void output_histogram(bucket * histogram){
	int i; 
	long long total_cnt = 0;
	for(i=0; i< num_buckets; i++) {
		if(i%5 == 0) /* we print 5 buckets in a row */
			printf("\n%02d: ", i);
		printf("%15lld ", histogram[i].d_cnt);
		total_cnt += histogram[i].d_cnt;
	  	/* we also want to make sure the total distance count is correct */
		if(i == num_buckets - 1)	
			printf("\n T:%lld \n", total_cnt);
		else printf("| ");
	}
}



void cuda_eror_check(cudaError_t errors, const char out[]){
	if (errors != cudaSuccess)
    {
        printf("There is something wrong with cuda %s, %s, \n", out, cudaGetErrorString(errors));
        exit(EXIT_FAILURE);
    }
}

void report_diff(bucket *histogram_CPU, bucket *histogram_GPU){
	//TODO
	printf("Histogram Difference:\n");
    for(int i = 0; i < num_buckets; i++) {
        if(i%5 == 0) /* we print 5 buckets in a row */
            printf("\n%02d: ", i);
        printf("%15lld ", (histogram_GPU[i].d_cnt - histogram_CPU[i].d_cnt));
        if(i != num_buckets - 1)
            printf("| ");
    }
    printf("\n\n");
}


int main(int argc, char **argv)
{
	int i;

	if (argc != 4 ) {
		printf("Error: Invalid Input, Please input all three parameters: {#of_samples} {bucket_width} {block_size}\n\n");
		exit(0);
	}

	PDH_acnt = atoi(argv[1]);
	PDH_res	 = atof(argv[2]);
	//printf("args are %d and %f\n", PDH_acnt, PDH_res);

	dim3 threads(atoi(argv[3]));
	dim3 grid(ceil((float)PDH_acnt/threads.x));

	num_buckets = (int)(BOX_SIZE * 1.732 / PDH_res) + 1;
	histogram = (bucket *)malloc(sizeof(bucket)*num_buckets);

	atom_list = (atom *)malloc(sizeof(atom)*PDH_acnt);

	
	srand(1);
	/* generate data following a uniform distribution */
	for(i = 0;  i < PDH_acnt; i++) {
		atom_list[i].x_pos = ((double)(rand()) / RAND_MAX) * BOX_SIZE;
		atom_list[i].y_pos = ((double)(rand()) / RAND_MAX) * BOX_SIZE;
		atom_list[i].z_pos = ((double)(rand()) / RAND_MAX) * BOX_SIZE;
	}

	bucket *device_histogram = NULL;
	atom *device_atomList = NULL;

	bucket *device_histogram1 = NULL;
	atom *device_atomList1 = NULL;

	size_t histogram_size = sizeof(bucket) * num_buckets;
	size_t atom_size = sizeof(atom)*PDH_acnt;


	/*Data Copy for Old method*/
	cudaError_t histogram_error =  cudaMalloc((void**) &device_histogram, histogram_size);
	cuda_eror_check(histogram_error,"Error during cudaMalloc for Histogram");
	cudaError_t atom_list_error =  cudaMalloc((void**) &device_atomList, atom_size);
	cuda_eror_check(atom_list_error,"Error during cudaMalloc for atom list");

	cudaError_t histogram_memcpy_error = cudaMemcpy(device_histogram, histogram, histogram_size, cudaMemcpyHostToDevice);
	cuda_eror_check(histogram_memcpy_error,"Error during cudaMemcpy for Histogram");

	cudaError_t atom_list_memcpy_error = cudaMemcpy(device_atomList, atom_list, atom_size, cudaMemcpyHostToDevice);
	cuda_eror_check(atom_list_memcpy_error,"Error during cudaMemcpy for atom list");


	/*Data Copy for New method*/
	cudaError_t histogram_error1 =  cudaMalloc((void**) &device_histogram1, histogram_size);
	cuda_eror_check(histogram_error1,"Error during cudaMalloc for Histogram");
	cudaError_t atom_list_error1 =  cudaMalloc((void**) &device_atomList1, atom_size);
	cuda_eror_check(atom_list_error1,"Error during cudaMalloc for atom list");

	cudaError_t histogram_memcpy_error1 = cudaMemcpy(device_histogram1, histogram, histogram_size, cudaMemcpyHostToDevice);
	cuda_eror_check(histogram_memcpy_error1,"Error during cudaMemcpy for Histogram");

	cudaError_t atom_list_memcpy_error1 = cudaMemcpy(device_atomList1, atom_list, atom_size, cudaMemcpyHostToDevice);
	cuda_eror_check(atom_list_memcpy_error1,"Error during cudaMemcpy for atom list");
	

	printf("\n************************************** GPU Old **********************************************\n");

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	
	/* start counting time */
	gettimeofday(&startTime, &Idunno);
	/* call CPU single thread version to compute the histogram */
	PDH_baseline <<<grid, threads>>> (device_histogram, device_atomList, PDH_res, PDH_acnt);

	double temp_time_1 = report_running_time();
	

	cudaError_t cuda_memcpy_error = cudaMemcpy(histogram, device_histogram, histogram_size, cudaMemcpyDeviceToHost);
	cuda_eror_check(cuda_memcpy_error, "Error during cudaMemcpy for Histogram to Host");
	
	

	
	output_histogram(histogram);
	printf("\n\n\n************************************** GPU Algo 3**********************************************\n");

	gettimeofday(&startTime, &Idunno);

	/*Call GPU Code*/
	PDH2D_basline_algorithm3<<<grid,threads,num_buckets*sizeof(bucket)>>>(device_histogram1, device_atomList1, PDH_acnt, PDH_res, grid.x, num_buckets);

	double temp_time_2 = report_running_time(1);
	
	bucket *device_histogram_temp = (bucket *) malloc(sizeof(bucket) * num_buckets);
	// cudaMemcpy(device_histogram_temp, device_histogram, histogram_size, cudaMemcpyDeviceToHost);

	//TODO
	cudaError_t last_error = cudaGetLastError();
	cuda_eror_check(last_error, "Check for last error by cudaGetLastError");
	/* check the total running time */ 
	
	

	cudaError_t cuda_memcpy_error1 = cudaMemcpy(device_histogram_temp, device_histogram, histogram_size, cudaMemcpyDeviceToHost);
	cuda_eror_check(cuda_memcpy_error1, "Error during cudaMemcpy for Histogram to Host");
	
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	
	
	/* print out the histogram */
	output_histogram(device_histogram_temp);

	printf("\n**********Time to generate:: %0.5f ms*********** \n\n", elapsedTime);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	/*
	printf("\n\n********************************** Difference **************************************************\n");
	printf("The Difference between Old and New GPU Time : %lf\n", temp_time_2 - temp_time_1);
	report_diff(histogram, device_histogram_temp);
	*/

	cuda_eror_check(cudaFree(device_histogram), "Free Device Historgram");
	cuda_eror_check(cudaFree(device_atomList),"Free Device Atom List");

	cuda_eror_check(cudaFree(device_histogram1), "Free Device Historgram");
	cuda_eror_check(cudaFree(device_atomList1),"Free Device Atom List");
	
	cudaFree(histogram);
	cudaFree(atom_list);
	cudaFree(device_histogram_temp);

	cuda_eror_check(cudaDeviceReset(), "Reset the Device");

	return 0;
}


