#include <stdio.h>
#include <iostream>
#include <algorithm>
#include <fstream>
#include <random>
#include <chrono>
#include "include/preprocessor.hpp"
#include "include/csv.hpp"
#include "include/cudadebug.cuh"
#include "include/encoding.cuh"
#include "include/kernels.cuh"
#include <cublas_v2.h>
// #include <cuda_runtime.h>

// #define USE_DOT_SIMILARITY
#define USE_COS_SIMILARITY

// #define USE_DOT_ENCODING
#define USE_LVID_ENCODING

int main(int argc, char* argv[]) {
    fclose(fopen("cluster_results.csv", "w"));  //clear file first
    fclose(fopen("train_labels.csv", "w"));  //clear file first
    // ./main [TRAIN dataset path] [DIM] [ITER] [Q]
    // Example:
    // ./main datasets/UCIHAR/UCIHAR_train.choir_dat 10000 20 100
    int nFeatures_train, nClasses_train;  // nFeatures is same as x_train[0].size()
    std::vector<std::vector<float>> x_train;
    std::vector<int> y_train;

    // readChoirDat(argv[1], nFeatures_train, nClasses_train, x_train, y_train);
    readFCPSTrainDat(argv[1], nFeatures_train, nClasses_train, x_train, y_train);

    l2norm(x_train);
    std::vector<float> x_train_flat = flatten(x_train);

    // base_creation: linear
    int dim = atoi(argv[2]);
    int iter_num = atoi(argv[3]);
    int Q = atoi(argv[4]); // nLevel

    // int K = atoi(argv[5]);
    int K = nClasses_train;

    int train_set_num = x_train.size();
    int base_size = nFeatures_train * dim;
    int train_encode_size = train_set_num * dim;

    std::cout << train_set_num << " " << nFeatures_train << std::endl;

    // K equals to the number of classes

    // generate bases
    std::vector<float> bases;  // flattened
    std::vector<float> base_v1(dim/2, 1);
    std::vector<float> base_v2(dim/2, -1);
    base_v1.insert(base_v1.end(), base_v2.begin(), base_v2.end());
    // obtain a time-based seed
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    for(int i = 0 ; i < nFeatures_train; i++) {
        std::shuffle(base_v1.begin(), base_v1.end(), std::default_random_engine(seed));
        bases.insert(bases.end(), base_v1.begin(), base_v1.end());
    }

    // generate level
    std::vector<float> level_base(base_v1);
    std::vector<float> level_hvs;
    for (int q = 0; q <= Q; ++q) {
        int flip = (int) (q/float(Q) * dim) / 2;
        std::vector<float> level_hv(level_base);
        // + flip will transform (flip) number of elements
        std::transform(level_hv.begin(), level_hv.begin() + flip, level_hv.begin(), bind2nd(std::multiplies<float>(), -1)); 
        level_hvs.insert(level_hvs.end(), level_hv.begin(), level_hv.end());
    }

    // generate id
    std::shuffle(level_base.begin(), level_base.end(), std::default_random_engine(seed));  // use this as id_base
    std::vector<float> id_hvs(level_base);  // f=0
    for (int f = 1; f < nFeatures_train; ++f) {
        std::rotate(level_base.begin(), level_base.begin() + 1, level_base.end());
        id_hvs.insert(id_hvs.end(), level_base.begin(), level_base.end());
    }

    // rand gen K number of 0 ~ train_set_num - 1
    std::vector<int> rand_weights(train_set_num);
    std::iota(std::begin(rand_weights), std::end(rand_weights), 0);                                                                                        
    std::shuffle(rand_weights.begin(), rand_weights.end(), std::default_random_engine(seed));  // random order
    rand_weights.resize(K);

    int* cluster_results;
    cluster_results = (int*)malloc(train_set_num * sizeof(int));

    //////////////////////////////////////////////////////////////////////////
    // GPU LOAD
    float* d_bases = NULL;
    float* d_x_train = NULL;
    float* d_hvs_train = NULL;

#ifdef USE_COS_SIMILARITY
    float* d_train_norm = NULL;
    float* d_weights_norm = NULL;
#endif
    HANDLE_ERROR(cudaMalloc((void **)&d_bases, base_size * sizeof(float)));
    HANDLE_ERROR(cudaMalloc((void **)&d_x_train, x_train_flat.size() * sizeof(float)));
    HANDLE_ERROR(cudaMalloc((void **)&d_hvs_train, 4 * train_encode_size * sizeof(float)));
    
#ifdef USE_COS_SIMILARITY
    HANDLE_ERROR(cudaMalloc((void **)&d_train_norm, train_set_num * sizeof(float)));
    HANDLE_ERROR(cudaMalloc((void **)&d_weights_norm, K * sizeof(float)));
#endif

    HANDLE_ERROR(cudaMemcpy(d_bases, bases.data(), base_size * sizeof(float), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_x_train, x_train_flat.data(), x_train_flat.size() * sizeof(float), cudaMemcpyHostToDevice));

    //id level
    float* d_level_hvs = NULL;
    float* d_id_hvs = NULL;
    HANDLE_ERROR(cudaMalloc((void **)&d_level_hvs, level_hvs.size() * sizeof(float)));
    HANDLE_ERROR(cudaMalloc((void **)&d_id_hvs, id_hvs.size() * sizeof(float)));
    HANDLE_ERROR(cudaMemcpy(d_level_hvs, level_hvs.data(), level_hvs.size() * sizeof(float), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_id_hvs, id_hvs.data(), id_hvs.size() * sizeof(float), cudaMemcpyHostToDevice));


    std::cout << "class: " << nClasses_train << std::endl;
    float* d_weights = NULL;
    HANDLE_ERROR(cudaMalloc((void **)&d_weights, K * dim * sizeof(float)));

    float* d_guess_table = NULL;
    int* d_y_pred = NULL;
    HANDLE_ERROR(cudaMalloc((void **)&d_guess_table, K * train_set_num * sizeof(float)));
    HANDLE_ERROR(cudaMalloc((void **)&d_y_pred, train_set_num * sizeof(int)));

    cudaEvent_t start, stop1, stop2;
    cudaError_t err = cudaSuccess;

    cudaEventCreate(&start);
    cudaEventCreate(&stop1);
    cudaEventCreate(&stop2);

    cublasHandle_t handle;
    cublasCreate(&handle);
    const float alpha = 1;
    const float beta = 0;

    cudaEventRecord(start);
    printf("Starting Encoding Stage...\n");

#ifdef USE_DOT_ENCODING
    // Encode stage: Linear
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
               dim, train_set_num, nFeatures_train, 
               &alpha, d_bases, dim, 
               d_x_train, nFeatures_train, &beta, 
               d_hvs_train, dim);
    cudaThreadSynchronize();
#endif

#ifdef USE_LVID_ENCODING
    dim3 encodeblocksTrain((dim + N_THREADS - 1) / N_THREADS, train_set_num);
    dim3 encodeTPB(N_THREADS, 1, 1);

    int level_stride = dim * 4;
    int id_stride = dim * 4;
    int fm_stride = nFeatures_train * 4;

    encodeLevelId<<<encodeblocksTrain, encodeTPB>>>(d_level_hvs, d_id_hvs, d_x_train, d_hvs_train, level_stride, 
                                                id_stride, fm_stride, train_set_num, nFeatures_train, Q, dim);
#endif

    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch kernel (error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Initialize class HVs
    for (int ii = 0; ii < K; ++ii) {
        HANDLE_ERROR(cudaMemcpy(d_weights + ii * dim, d_hvs_train + rand_weights[ii] * dim, dim * sizeof(float), cudaMemcpyDeviceToDevice));
    }

    cudaEventRecord(stop1);
    cudaEventSynchronize(stop1);

    // Clustering stage
    // make_guess and create guess table
    printf("Clustering stage...\n");

#ifdef USE_COS_SIMILARITY
    normMatRow<<<(train_set_num + N_THREADS - 1) / N_THREADS, N_THREADS>>>(d_train_norm, d_hvs_train, train_set_num, dim);
#endif

    for (int iter = 0; iter < iter_num; ++iter) {  // Retraining, Different from max_iter
        cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, 
                    K, train_set_num, dim, 
                    &alpha, d_weights, dim, 
                    d_hvs_train, dim, &beta, 
                    d_guess_table, K);   // np.dot
        cudaThreadSynchronize();

#ifdef USE_COS_SIMILARITY
        normMatRow<<<(K + N_THREADS - 1) / N_THREADS, N_THREADS>>>(d_weights_norm, d_weights, K, dim);

        guessVecGenCompareCosine<<<(train_set_num + N_THREADS - 1)/N_THREADS, N_THREADS>>>(d_y_pred, d_weights_norm, d_train_norm, d_guess_table, train_set_num, K);
#endif

#ifdef USE_DOT_SIMILARITY
        guessVecGenCompareDot<<<(train_set_num + N_THREADS - 1)/N_THREADS, N_THREADS>>>(d_y_pred, d_guess_table, train_set_num, K);
#endif

        updateClassHV<<<(dim + N_THREADS - 1) / N_THREADS, N_THREADS>>>(d_hvs_train, d_weights, d_y_pred, train_set_num, dim);
    }

    HANDLE_ERROR(cudaMemcpy(cluster_results, d_y_pred, train_set_num * sizeof(int), cudaMemcpyDeviceToHost));
    cudaEventRecord(stop2);
    cudaEventSynchronize(stop2);

    cublasDestroy(handle);
    HANDLE_ERROR(cudaFree(d_bases));
    HANDLE_ERROR(cudaFree(d_x_train));
    HANDLE_ERROR(cudaFree(d_hvs_train));
    HANDLE_ERROR(cudaFree(d_weights));

    HANDLE_ERROR(cudaFree(d_guess_table));
    HANDLE_ERROR(cudaFree(d_y_pred));

#ifdef USE_COS_SIMILARITY
    HANDLE_ERROR(cudaFree(d_train_norm));
    HANDLE_ERROR(cudaFree(d_weights_norm));
#endif

    HANDLE_ERROR(cudaFree(d_level_hvs));
    HANDLE_ERROR(cudaFree(d_id_hvs));

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop1);
    printf("GPU Execution time (Encoding): %f\n", milliseconds);
    cudaEventElapsedTime(&milliseconds, stop1, stop2);
    printf("GPU Execution time (Clustering): %f\n", milliseconds);

    // write file
    std::ofstream out("cluster_results.csv");
    for (int idx = 0; idx < train_set_num; ++idx) {
        out << cluster_results[idx] <<',';
    }
    out << '\n';
    
    std::ofstream out_gnd("train_labels.csv");
    for (auto& data : y_train) {
        out_gnd << data <<',';
    }
    out_gnd << '\n';
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop1);
    cudaEventDestroy(stop2);

    free(cluster_results);

    return 0;
}
