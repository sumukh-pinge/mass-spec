#ifndef __KERNELS_CUH__
#define __KERNELS_CUH__

#define N_THREADS 1024


__global__ void normMatRow(float* result, float* inputMat, int setNum, int colNum);
__global__ void cosineSimilarityVec(float* result, float* norm_1, int colNum, float* norm_2_const, int dataidx);
__global__ void guessVecGenCompareCosine(int* y_pred, float* weights_norm, float* data_norm,
                                         const float * guess_table, const int setNum, const int n_class);
__global__ void guessVecGenCompareDot(int* y_pred, const float * guess_table, const int setNum, const int n_class);

#endif
