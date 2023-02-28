#include "../include/kernels.cuh"
#include <thrust/device_vector.h>
#include <thrust/transform_reduce.h>

struct square { __host__ __device__ float operator()(float x) { return x * x; } };

__global__ void normMatRow(float* result, float* inputMat, int setNum, int colNum) {
    for (int rowNum = blockIdx.x * blockDim.x + threadIdx.x; 
        rowNum < setNum; 
        rowNum += blockDim.x * gridDim.x)
    {
        // result[rowNum] = normf(colNum, inputMat + rowNum * colNum);
        result[rowNum] = sqrt(thrust::transform_reduce(thrust::device, inputMat + rowNum * colNum, inputMat + (rowNum + 1) * colNum, square(), 0.0f, thrust::plus<float>()));
    }
}

__global__ void cosineSimilarityVec(float* result, float* norm_1, int colNum, float* norm_2_const, int dataidx) {
    float norm_2 = norm_2_const[dataidx];
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < colNum)
        result[idx] = result[idx] / (norm_1[idx] * norm_2);
}

__global__
void guessVecGenCompareCosine(int* y_pred, float* weights_norm, float* data_norm,
    const float * guess_table, const int setNum, const int n_class) {

    int rowNum = threadIdx.x + blockDim.x * blockIdx.x;
    float norm_2 = data_norm[rowNum];

    if (rowNum < setNum) {
        float max_value = guess_table[n_class * rowNum + 0] / (weights_norm[0] * norm_2);
        // float max_value = guess_table[n_class * rowNum + 0]; 
        int max_idx = 0;
        for (int j = 1; j < n_class; j++){
            float val_to_compare = guess_table[n_class * rowNum + j] / (weights_norm[j] * norm_2);
            if (max_value < val_to_compare) {
                max_value = val_to_compare;
                max_idx = j;
            }
        }
        y_pred[rowNum] = max_idx;
    }
}

__global__
void guessVecGenCompareDot(int* y_pred,
    const float * guess_table, const int setNum, const int n_class) {

    int rowNum = threadIdx.x + blockDim.x * blockIdx.x;
    if (rowNum < setNum) {
        float max_value = guess_table[n_class * rowNum + 0];
        // float max_value = guess_table[n_class * rowNum + 0];
        int max_idx = 0;
        for (int j = 1; j < n_class; j++){
            float val_to_compare = guess_table[n_class * rowNum + j];
            if (max_value < val_to_compare) {
                max_value = val_to_compare;
                max_idx = j;
            }
        }
        y_pred[rowNum] = max_idx;
    }
}

