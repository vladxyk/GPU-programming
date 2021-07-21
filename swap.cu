#include <iostream>
#include <cuda.h>
#include <cstdio>
#include <sys/time.h>
#include <cublas_v2.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/swap.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>

#define CUDA_CHECK_RETURN(value) {\
cudaError_t _m_cudaStat = value;\
if (_m_cudaStat != cudaSuccess) {\
  fprintf(stderr, "Error %s at line %d in file %s\n",\
          cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);\
  exit(1);\
}\
}

using namespace std;
//a[1 2 3 4]
//b[5 6 7 8]
//swap
//a[5 6 7 8]
//b[1 2 3 4]

__global__ void swap(float *a, float *b, int vector_size){
    
	int indx = blockIdx.x * blockDim.x + threadIdx.x;

  float k = a[indx];
  a[indx] = b[indx];
  b[indx] = k;
}

float thrust_swap(int vector_size){

    thrust::host_vector<float> hA(vector_size);
    thrust::host_vector<float> hB(vector_size);

    for(int i = 0; i < vector_size; i++)
    {
        hA[i] = rand()%5;
        hB[i] = rand()%5;
    }

    thrust::device_vector<float> dA = hA;
    thrust::device_vector<float> dB = hB;

    cout << "THRUST_SWAP" << endl;
    /*
    cout << "vector A before swap : ";
    for (int i = 0; i < vector_size; i++){
        cout << hA[i] << " ";
    }

    cout << endl;

    cout << "vector B before swap : ";
    for (int i = 0; i < vector_size; i++){
      cout << hB[i] << " ";
    }
    cout << endl;
    */
    float ThrustTime;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    thrust::swap(dA, dB);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&ThrustTime, start, stop);

    thrust::copy(dA.begin(), dA.end(), hA.begin());
    thrust::copy(dB.begin(), dB.end(), hB.begin());

    /*
    cout << "vector A after swap: ";
    for (int i = 0; i < vector_size; i++){
        cout << hA[i] << " ";
    }

    cout << endl;

    cout << "vector B after swap: ";
    for (int i = 0; i < vector_size; i++){
      cout << hB[i] << " ";
    }
    cout << endl;
    */
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return ThrustTime;

}
float blas_swap(int vector_size){

    float *hA = new float[vector_size];
    float *hB = new float[vector_size];

    for(int i = 0; i < vector_size; i++)
    {
        hA[i] = rand()%5;
        hB[i] = rand()%5;
    }

    float *dA, *dB;
    
    CUDA_CHECK_RETURN(cudaMalloc(&dA, sizeof(float) * vector_size));
    CUDA_CHECK_RETURN(cudaMalloc(&dB, sizeof(float) * vector_size));

    CUDA_CHECK_RETURN(cudaMemcpy(dA, hA, sizeof(float) * vector_size, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(dB, hB, sizeof(float) * vector_size, cudaMemcpyHostToDevice));

    cout << "CUBLAS_SWAP" << endl;
    /*
    cout << "vector A before swap: ";
    for (int i = 0; i < vector_size; i++){
        cout << hA[i] << " ";
    }

    cout << endl;

    cout << "vector B before swap : ";
    for (int i = 0; i < vector_size; i++){
      cout << hB[i] << " ";
    }
    cout << endl;
    */
    cublasHandle_t handle;
    cublasCreate(&handle);

    float CublasTime;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    cublasSswap(handle, vector_size, dA, 1, dB, 1);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    CUDA_CHECK_RETURN(cudaGetLastError());

    cudaEventElapsedTime(&CublasTime, start, stop);

    CUDA_CHECK_RETURN(cudaMemcpy(hA, dA, sizeof(float) * vector_size, cudaMemcpyDeviceToHost));
    CUDA_CHECK_RETURN(cudaMemcpy(hB, dB, sizeof(float) * vector_size, cudaMemcpyDeviceToHost));
    
    /*
    cout << "vector A after swap : ";
    for (int i = 0; i < vector_size; i++){
        cout << hA[i] << " ";
    }

    cout << endl;

    cout << "vector B after swap : ";
    for (int i = 0; i < vector_size; i++){
      cout << hB[i] << " ";
    }
    cout << endl;
    */
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cublasDestroy(handle);
    delete [] hA;
    delete [] hB;
    cudaFree(dA);
    cudaFree(dB);

    return CublasTime;
}

int main(int argc, char *argv[])
{
    cout << "1 arg - vector_size, 2 arg - block_size" << endl << endl;

    int vector_size = atoi(argv[1]);
    int block_size = atoi(argv[2]);
        
    srand(time(NULL));
    
    float *hA = new float[vector_size];
    float *hB = new float[vector_size];

    for(int i = 0; i < vector_size; i++)
    {
        hA[i] = rand()%5;
        hB[i] = rand()%5; 
    }

    float *dA, *dB;
    
    CUDA_CHECK_RETURN(cudaMalloc(&dA, sizeof(float) * vector_size));
    CUDA_CHECK_RETURN(cudaMalloc(&dB, sizeof(float) * vector_size));

    CUDA_CHECK_RETURN(cudaMemcpy(dA, hA, sizeof(float) * vector_size, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(dB, hB, sizeof(float) * vector_size, cudaMemcpyHostToDevice));

    cout << "CUDA_SWAP" << endl;
   /*
    cout << "vector A before swap : ";
    for (int i = 0; i < vector_size; i++){
        cout << hA[i] << " ";
    }

    cout << endl;

    cout << "vector B before swap : ";
    for (int i = 0; i < vector_size; i++){
      cout << hB[i] << " ";
    }
    cout << endl;
    */
    int num_blocks = (int)ceil((float)vector_size / block_size);
    
    float elapsedTime;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    swap <<<num_blocks, block_size>>> (dA, dB, vector_size);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    CUDA_CHECK_RETURN(cudaGetLastError());

    cudaEventElapsedTime(&elapsedTime, start, stop);

    CUDA_CHECK_RETURN(cudaMemcpy(hA, dA, sizeof(float) * vector_size, cudaMemcpyDeviceToHost));
    CUDA_CHECK_RETURN(cudaMemcpy(hB, dB, sizeof(float) * vector_size, cudaMemcpyDeviceToHost));
  
    //cout << "CUDA_COPY" << endl;
  /* 
    cout << "vector A after swap : ";
    for (int i = 0; i < vector_size; i++){
        cout << hA[i] << " ";
    }

    cout << endl;

    cout << "vector B after swap : ";
    for (int i = 0; i < vector_size; i++){
      cout << hB[i] << " ";
    }
    cout << endl;
   */
    cout << "Cuda_Time = " << elapsedTime << endl;
    cout << endl;
    
    float cublas = blas_swap(vector_size);
    cout << "Cublas_Time = " << cublas << endl;
    cout << endl;
  

    float thrust = thrust_swap(vector_size);
    cout << "Thrust_Time = " << thrust << endl;
    cout << endl;
  
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    delete [] hA;
    delete [] hB;
    cudaFree(dA);
    cudaFree(dB);
}
