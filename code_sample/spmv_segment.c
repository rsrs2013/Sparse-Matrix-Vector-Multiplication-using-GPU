#include "genresult.cuh"
#include <sys/time.h>



__device__ float segmented_scan(const int lane,const int* rows, float* ptrs){
 
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


__global__ void putProduct_kernel(int nz, int *rIndices, int *cIndices, float *values, int M, int N, float *vector, 
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

            //Indicator variable to know, if the value needs to be accumulated from the previous warp;
            unsigned int warp_open = 0;
            if(warpid!=0){
              
               if(rows[threadIdx.x] == rows[warp_first_threadId-1]){
                  
                  warp_open = 1;

               }

            }


            float val = segmented_scan(lane,rows,ptrs);
            row = rows[threadIdx.x];

           
            __syncthreads();

            //The last thread in each warp writes the partial results;

            if(threadIdx.x == warp_last_threadId){
               
               ptrs[warpid] = val;
               rows[warpid] = rows[warp_last_threadId];

            } 
            
            __syncthreads();

            //The first warp scans the per-warp results
            if(warpid == 0){
                 
                 segmented_scan(lane,rows,ptrs);

            }

            __syncthreads();

            //
            if(warpid!=0 && warp_open == 1){
               
               val = val+ptrs[warpid-1];

            }

            __syncthreads();

            ptrs[threadIdx.x] = val;
            rows[threadIdx.x] = row;

            __syncthreads();


            if(data_id == nz-1){

            	//if the thread is the last thread;
               
               atomicAdd(&result[rows[threadIdx.x]],ptrs[threadIdx.x]);

            }else if(threadIdx.x != blockDim.x-1){

                //if the thread is not the last thread in a block,and next row is different from thread's row
                if(rows[threadIdx.x] != rows[threadIdx.x+1]){
                   
                   atomicAdd(&result[rows[threadIdx.x]],ptrs[threadIdx.x]);                   
                }
            
            }
            else{

                  atomicAdd(&result[rows[threadIdx.x]],ptrs[threadIdx.x]);

            }

             
        }
    }




}


typedef struct coordFormat{

    int row;
    int col;
    float val;

}coordFormat;

int compareFunction(const void* a, const void* b){

     int l = ((coordFormat*)a)->row;
     int r = ((coordFormat*)b)->row;
     return (l-r);
}



void getMulScan(MatrixInfo * mat, MatrixInfo * vec, MatrixInfo * res, int blockSize, int blockNum){
    /*Allocate things...*/
    int number_of_non_zeros = mat->nz;
    int *row_indices = mat->rIndex;
    int *column_indices = mat->cIndex;
    float *values = mat->val;
    int M = mat->M;
    int N = mat->N;
    float *vector = vec->val;
    float *result = res->val;  

    /*Sorting the rows in the order*/
    coordFormat* arrayOfStructs = (coordFormat*)malloc(sizeof(coordFormat)*mat->nz);

    for(int i=0;i<mat->nz;i++){
        
        arrayOfStructs[i].row = row_indices[i];
        arrayOfStructs[i].col = column_indices[i];
        arrayOfStructs[i].val = values[i];

    }

    //apply the inbuilt quicksort function;
    qsort(arrayOfStructs,mat->nz,sizeof(coordFormat),compareFunction);

    //get the sorted rows back into the row_indices and column indices and values;

    for(int i=0;i<mat->nz;i++){

    	row_indices[i] = arrayOfStructs[i].row;
    	column_indices[i] = arrayOfStructs[i].col;
    	values[i] = arrayOfStructs[i].val;
    }

    printf("\nGPU Code");
    printf("\nBlock Size : %lu, Number of Blocks : %lu, nz : %lu\n",blockSize,blockNum,mat->nz);  
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

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    /*Invoke kernel(s)*/

    putProduct_kernel<<<blockSize,blockNum>>>(number_of_non_zeros,d_rIndices,d_cIndices,d_values,M,N,d_vector,d_result);

    cudaDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    printf("Segmented Kernel Time: %lu micro-seconds\n", 1000000 * (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1000);

    cudaMemcpy(result,d_result,sizeof(float)*M, cudaMemcpyDeviceToHost);
    res->val = result; 

    /*Deallocate, please*/
    cudaFree(d_rIndices);
    cudaFree(d_cIndices);
    cudaFree(d_values);
    cudaFree(d_vector);
    cudaFree(d_result);
}
