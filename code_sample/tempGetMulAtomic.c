__global__ void getMulAtomic_kernel(int nnz,int* coord_row,int* coord_col,
    float* A,float* x,float* y){
    /* This is just an example empty kernel, you don't have to use the same kernel
 * name*/

  int thread_id = (blockDim.x * blockIdx.x) + threadIdx.x;
  int thread_num = blockDim.x * gridDim.x;
  int iter = nnz % thread_num ? nnz/thread_num+1:nnz/thread_num;

  for(int i =0; i<iter; i++){

     int dataid = thread_id + i*thread_num;
     if(dataid < nnz){
        
        float data = A[dataid];
        int row  = coord_row[dataid];
        int col = coord_col[dataid];
        float temp = data*x[col];
        atomicAdd(&y[row],temp);

     }

  }

    
}

void getMulAtomic(MatrixInfo * mat, MatrixInfo * vec, MatrixInfo * res, int blockSize, int blockNum){

    /*Allocate here...*/
     
    int nz = mat->nz; 
    //float *y = malloc(sizeof(float)*mat->N);
    //memset(y,0,sizeof(float)*mat->N);
    memset(res->val,0,sizeof(float)*res->M);

    int *d_coord_row, *d_coord_col;
    float *d_A, *d_x, *d_y;
    
    cudaMalloc((void **)&d_coord_row,sizeof(int)*nz);
    cudaMalloc((void **)&d_coord_col,sizeof(int)*nz);
    cudaMalloc((void **)&d_A,sizeof(float)*nz);
    cudaMalloc((void **)&d_x,sizeof(float)*mat->N);
    cudaMalloc((void **)&d_y,sizeof(float)*mat->M);

    cudaMemset(d_y,0,sizeof(float)*mat->M);

    cudaMemcpy(d_coord_row,mat->rIndex,sizeof(int)*nz,cudaMemcpyHostToDevice);
    cudaMemcpy(d_coord_col,mat->cIndex,sizeof(int)*nz,cudaMemcpyHostToDevice);
    cudaMemcpy(d_A,mat->val,sizeof(float)*nz,cudaMemcpyHostToDevice);
    cudaMemcpy(d_x,vec->val,sizeof(float)*N,cudaMemcpyHostToDevice);
    //cudaMemcpy(d_y,res->val,sizeof(float)*mat->M,cudaMemcpyHostToDevice);

    
		/* Sample timing code */
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    /*Invoke kernels...*/
    printf("Invoking the kernel\n");
    getMulAtomic<<<blockNum,blockSize>>>(nz,d_coord_row,d_coord_col,d_A,d_x,d_y); 


    cudaDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    printf("Atomic Kernel Time: %lu micro-seconds\n", 1000000 * (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1000);
    /*Deallocate.*/
    cudaMemcpy(res->val,d_y,sizeof(float)*M,cudaMemcpyDeviceToHost);
    printf("[");
    for(int i =0;i<res->M;i++){
      printf("%f,",res->val[i]);

    }
    printf("]");

    //Clean up
    cudaFree(d_coord_row);
    cudaFree(d_coord_col);
    cudaFree(d_A);
    cudaFree(d_x);
    cudaFree(d_y);

}