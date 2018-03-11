#include "genresult.cuh"
#include <sys/time.h>

/* Put your own kernel(s) here*/

__device__ float segmented_scan_design(const int lane,const int* rows, float* ptrs){

  
   if(lane>=1 && rows[threadIdx.x] ==  rows[threadIdx.x-1])
      ptrs[threadIdx.x]+=ptrs[threadIdx.x-1];

    if(lane>=2 && rows[threadIdx.x] ==  rows[threadIdx.x-2])
      ptrs[threadIdx.x]+=ptrs[threadIdx.x-2];

    if(lane>=4 && rows[threadIdx.x] ==  rows[threadIdx.x-4])
      ptrs[threadIdx.x]+=ptrs[threadIdx.x-4];

    if(lane>=8 && rows[threadIdx.x] ==  rows[threadIdx.x-8])
      ptrs[threadIdx.x]+=ptrs[threadIdx.x-8];

    if(lane>=16 && rows[threadIdx.x] ==  rows[threadIdx.x-16])
      ptrs[threadIdx.x]+=ptrs[threadIdx.x-16];
    
    return ptrs[threadIdx.x];

}

__global__ void putProduct_kernel_design(int nz, int *rIndices, int *cIndices, float *values, int M, int N, float *vector, 
	float *result){
    /*Put your kernel(s) implementation here, you don't have to use exactly the
 * same kernel name */

	//create a shared memory;
    __shared__ int rows[1024];
    __shared__ float ptrs[1024];
    
 
    int thread_id = threadIdx.x + (blockIdx.x *blockDim.x);
    int total_number_of_threads = blockDim.x * gridDim.x;
    int iteration = nz % total_number_of_threads ? nz/total_number_of_threads + 1 : nz/total_number_of_threads;
    

    for( int i = 0 ; i < iteration ; i++ ) {

        int data_id = thread_id + i* total_number_of_threads;
        if( data_id < nz ) {

            float data = values[data_id];
            int row = rIndices[data_id];
            int column = cIndices[data_id];
            float multiplication_value = data * vector[column];
            //copy values into the shared memory;
            ptrs[threadIdx.x] = multiplication_value;
            rows[threadIdx.x] = row;
            //atomicAdd(&result[row],multiplication_value);
            __syncthreads();
            
            //warp details of the thread;
            unsigned int warpid = threadIdx.x>>5;
            unsigned int warp_first_threadId = warpid<<5;
            unsigned int warp_last_threadId = warp_first_threadId+31;
            const unsigned int lane = threadIdx.x%32;

           
         
            float val = segmented_scan_design(lane,rows,ptrs);
            row = rows[threadIdx.x];
            __syncthreads();
            
            /*if the next value of the row different from this index row value, 
            then need to copy in the result matrix;*/
            if((threadIdx.x == warp_last_threadId) || (rows[threadIdx.x] != rows[threadIdx.x+1])){

                  //do an atomic add;
            	  atomicAdd(&result[rows[threadIdx.x]],ptrs[threadIdx.x]);

            }
            
        }
}

}

typedef struct cFormat{

    int row;
    int col;
    float val;

}cFormat;


int comparisonFunction(const void* a, const void* b){

     int l = ((cFormat*)a)->row;
     int r = ((cFormat*)b)->row;
     return (l-r);
}


void getMulDesign(MatrixInfo * mat, MatrixInfo * vec, MatrixInfo * res, int blockSize, int blockNum){

    /*Allocate*/
    int number_of_non_zeros =  mat->nz;
    int *row_indices = mat->rIndex;
    int *column_indices = mat->cIndex;
    float *values = mat->val;
    int M = mat->M;
    int N = mat->N;
    float *vector = vec->val;
    float *result = res->val;  

    /*Sorting the rows in the order*/
    cFormat* arrayOfCoordinates = (cFormat*)malloc(sizeof(cFormat)*mat->nz);

    for(int i=0;i<mat->nz;i++){
        
        arrayOfCoordinates[i].row = row_indices[i];
        arrayOfCoordinates[i].col = column_indices[i];
        arrayOfCoordinates[i].val = values[i];

    }

     //apply the inbuilt quicksort function;
    qsort(arrayOfCoordinates,mat->nz,sizeof(cFormat),comparisonFunction);


     //get the sorted rows back into the row_indices and column indices and values;

    for(int i=0;i<mat->nz;i++){

    	row_indices[i] = arrayOfCoordinates[i].row;
    	column_indices[i] = arrayOfCoordinates[i].col;
    	values[i] = arrayOfCoordinates[i].val;
    }

     printf("\nGPU Code");
    printf("\nBlock Size : %lu, Number of Blocks : %lu, nz : %lu\n",blockSize,blockNum,number_of_non_zeros);
    
    /* Device copies of the required values */
    int   * d_rIndices;
    int   * d_cIndices;
    float *d_values;
    float *d_vector;
    float *d_result;

     /* Allocate values for the device copies */
    cudaMalloc((void**)&d_rIndices, sizeof(int)*number_of_non_zeros);
    cudaMalloc((void**)&d_cIndices, sizeof(int)*number_of_non_zeros);
    cudaMalloc((void**)&d_values, sizeof(float)*number_of_non_zeros);
    cudaMalloc((void**)&d_vector, sizeof(float)*N);
    cudaMalloc((void**)&d_result, sizeof(float)*M);
    
    /* Set all the result values to be zeros */
    cudaMemset(d_result, 0, sizeof(float)*M);
    
    /* Copying values from host to device */
    cudaMemcpy(d_rIndices,row_indices,sizeof(int)*number_of_non_zeros, cudaMemcpyHostToDevice);
    cudaMemcpy(d_cIndices,column_indices, sizeof(int)*number_of_non_zeros , cudaMemcpyHostToDevice);
    cudaMemcpy(d_values,values, sizeof(float)*number_of_non_zeros,cudaMemcpyHostToDevice);
    cudaMemcpy(d_vector,vector, sizeof(float)*N, cudaMemcpyHostToDevice);
    
    //int *numberLeftForEachRow = (int*)malloc(sizeof(int)*M);
    //memset(numberLeftForEachRow,0,sizeof(int)*M);
    //int index =0;

   
    struct timespec start, end;
    cudaDeviceSynchronize();

    
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    /*Your own magic here!*/
   
    putProduct_kernel_design<<<blockSize,blockNum>>>(number_of_non_zeros,d_rIndices,d_cIndices,d_values,M,N,d_vector,d_result);


    cudaDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    printf("Your Own Kernel Time: %lu micro-seconds\n", 1000000 * (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1000);

    /*Deallocate*/
    cudaMemcpy(result,d_result,sizeof(float)*M, cudaMemcpyDeviceToHost);
    res->val = result; 

    /*Deallocate, please*/
    cudaFree(d_rIndices);
    cudaFree(d_cIndices);
    cudaFree(d_values);
    cudaFree(d_vector);
    cudaFree(d_result);
}
