#include "../include/encoding.cuh"

__device__ float* get2df(float* p, const int x, int y, const int stride) {
        return (float*)((char*)p + x*stride) + y;
}

__global__ void encodeLevelId(
	float* level_hvs, float* id_hvs, float* feature_matrix, float* hv_matrix,
    int level_stride, int id_stride, int fm_stride, int N, int F, int Q, int D)
{
    const int sample_idx = blockIdx.y;
	if (sample_idx >= N)
        return;

    const int d = threadIdx.x + blockIdx.x * blockDim.x;
	if (d >= D)
        return;

	int f;
    float encoded_hv_e = 0.0;
#pragma unroll 1
    for (f = 0; f < F; ++f) {
        float v = *get2df(feature_matrix, sample_idx, f, fm_stride);
        encoded_hv_e += *get2df(level_hvs, (int)(v * Q), d, level_stride) * \
                        *get2df(id_hvs, f, d, id_stride);
    }
    // binarize? MAS?
    hv_matrix[sample_idx * D + d] = encoded_hv_e;
}

__global__ void updateClassHV(float* hv_matrix, float* weights, int* y_pred, int N, int D)
{
    const int d = threadIdx.x + blockIdx.x * blockDim.x;
	if (d >= D)
        return;
    
    const int MAX_CLASS = 50;  // Maximum supported Classes

    float sum_temp[MAX_CLASS] = {0, };
    // int class_cnt[MAX_CLASS] = {0, };

    for(int ii = 0; ii < N; ++ii) {
        int idx = y_pred[ii];
        sum_temp[idx] += hv_matrix[ii * D + d];  // 0~K-1
        // class_cnt[idx]++;
    }
    for(int ii = 0; ii < N; ++ii) {
        int idx = y_pred[ii];
        
        if (sum_temp[idx] == 0)
            weights[idx * D + d] = hv_matrix[20 * D + d];  //Randomize
        else
            weights[idx * D + d] = sum_temp[idx];
    }
}

