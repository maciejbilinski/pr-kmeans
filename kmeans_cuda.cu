#include <iostream>
#include <map>
#include <cmath>

#include "constants.h"
#include "point.h"
#include "common_func.h"

int THREADS_PER_BLOCK;

#define getIndex(row, column) row * DATA_DIMENSIONS + column

int getGridSize(int n){
    return (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
}

// inicjalizuje centroidy na podstawie losowo wybranych identyfikatorów znajdujących się w number
__global__ void initCentroids(int* i, int* numbers, float* centroids, float* data){
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (j < DATA_DIMENSIONS) {
        centroids[getIndex(*i, j)] = data[getIndex(numbers[*i], j)];
    }
}

// metoda `distance(Point*, Point*)` przepisana na działanie na GPU
__device__ void gpuDistance(float* p1, float* p2, float* output) {
    float sum = 0.0;
    for (int i = 0; i < DATA_DIMENSIONS; i++) {
        float diff = *(p1 + i) - *(p2 + i);
        sum += (diff * diff);
    }
    *output = std::sqrt(sum);
}

// metoda `gpuGetCentroid(Point, Point*)` przepisana na działanie na GPU
__device__ void gpuGetCentroid(float* p, float* centroids, int* output) {
    int closestCentroid = 0;
    float minDist;
    gpuDistance(p, centroids, &minDist);
    for (int j = 1; j < LABELS_NUMBER; j++) {
        float dist;
        gpuDistance(p, centroids + j * DATA_DIMENSIONS, &dist);
        if (dist < minDist) {
            minDist = dist;
            closestCentroid = j;
        }
    }
    *output = closestCentroid;
}

// wstępnie oblicza wartości centroidów
__global__ void calculateCentroids(float* centroids, float* data, float* newCentroids, int* clusterSizes){
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < DATA_LINES) {
        int cluster;
        gpuGetCentroid(data + i*DATA_DIMENSIONS, centroids, &cluster);
        for (int j = 0; j < DATA_DIMENSIONS; j++) {
            atomicAdd(&newCentroids[getIndex(cluster, j)], data[getIndex(i, j)]);
        }
        atomicAdd(&clusterSizes[cluster], 1);
    }
}

// dokańcza obliczanie wartości centroidów
__global__ void finishCalculateCentroids(float* newCentroids, int* clusterSizes){
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < LABELS_NUMBER) {
        if (clusterSizes[i] > 0) {
            for (int j = 0; j < DATA_DIMENSIONS; j++) {
                newCentroids[getIndex(i, j)] /= clusterSizes[i];
            }
        }
    }
}

// przepisuje wartości z `newCentroids` do `centroids`
__global__ void rewriteCentroids(int* i, float* newCentroids, float* centroids){
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < DATA_DIMENSIONS) {
        centroids[getIndex(*i, j)] = newCentroids[getIndex(*i, j)];
    }
}

__global__ void checkCentroidsChanged(float* centroids, float* newCentroids, bool* centroidsChanged) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < LABELS_NUMBER) {
        float dist;
        gpuDistance(centroids + i, newCentroids + i, &dist);
        if(dist > 0.0){
            *centroidsChanged = true;
        }
    }
}

void kMeansCUDA(Point* cpuCentroids, Point* cpuData, int MAX_ITERATIONS) {
    // kopiowanie danych na GPU
    float* centroids;
    float* data;
    cudaMalloc((void**)&centroids, LABELS_NUMBER * DATA_DIMENSIONS * sizeof(float));
    cudaMalloc((void**)&data, DATA_LINES * DATA_DIMENSIONS * sizeof(Point));
    for (int i = 0; i < DATA_LINES; i++) {
        cudaMemcpy(&data[i*DATA_DIMENSIONS], cpuData[i].values, DATA_DIMENSIONS * sizeof(float), cudaMemcpyHostToDevice);
    }

    // losowanie początkowych wartości
    int* cpuNumbers = getRandomDataIndexes();
    int* numbers;
    cudaMalloc((void**)&numbers, LABELS_NUMBER * sizeof(int));
    cudaMemcpy(numbers, cpuNumbers, LABELS_NUMBER * sizeof(int), cudaMemcpyHostToDevice);
    for (int i = 0; i < LABELS_NUMBER; i++) {
        int* x;
        cudaMalloc((void**)&x, sizeof(int));
        cudaMemcpy(x, &i, sizeof(int), cudaMemcpyHostToDevice);
        initCentroids<<<getGridSize(DATA_DIMENSIONS), THREADS_PER_BLOCK>>>(x, numbers, centroids, data);
        cudaDeviceSynchronize();
        cudaFree(x);
    }
    cudaFree(numbers);
    delete[] cpuNumbers;

    // inicjalizacja pamięci
    int* clusterSizes;
    float* newCentroids;
    cudaMalloc((void**)&clusterSizes, LABELS_NUMBER * sizeof(int));
    cudaMalloc((void**)&newCentroids, LABELS_NUMBER * DATA_DIMENSIONS * sizeof(float));

    int* clusterSizes0 = (int*)calloc(LABELS_NUMBER, sizeof(int));
    float* newCentroids0 = (float*)calloc(LABELS_NUMBER * DATA_DIMENSIONS, sizeof(float));

    // główna pętla algorytmu
    for (int iter = 0; iter < MAX_ITERATIONS; iter++) {
        if (VERBOSE)
            std::cout << "Iteration " << iter << std::endl;

        // czyszczenie pamięci - ustalanie wartości na zero
        cudaMemcpy(clusterSizes, clusterSizes0, LABELS_NUMBER * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(newCentroids, newCentroids0, LABELS_NUMBER * DATA_DIMENSIONS * sizeof(float), cudaMemcpyHostToDevice);

        // poszukiwanie nowych centroidów
        calculateCentroids<<<getGridSize(DATA_LINES), THREADS_PER_BLOCK>>>(centroids, data, newCentroids, clusterSizes);
        cudaDeviceSynchronize();
        finishCalculateCentroids<<<getGridSize(LABELS_NUMBER), THREADS_PER_BLOCK>>>(newCentroids, clusterSizes);
        cudaDeviceSynchronize();

        // opcjonalny mechanizm wczesnego zatrzymywania algorytmu
        if (EARLY_STOP) {
            bool centroidsChanged = false;
            bool* gpuCentroidsChanged;

            cudaMalloc(&gpuCentroidsChanged, sizeof(bool));
            cudaMemcpy(gpuCentroidsChanged, &centroidsChanged, sizeof(bool), cudaMemcpyHostToDevice);

            checkCentroidsChanged<<<getGridSize(LABELS_NUMBER), THREADS_PER_BLOCK>>>(centroids, newCentroids, gpuCentroidsChanged);
            cudaDeviceSynchronize();
            cudaMemcpy(&centroidsChanged, gpuCentroidsChanged, sizeof(bool), cudaMemcpyDeviceToHost);
            
            cudaFree(gpuCentroidsChanged);

            if (!centroidsChanged) {
                break;
            }
        }

        // przepisywanie nowych centroidów
        for (int i = 0; i < LABELS_NUMBER; i++) {
            int* x;
            cudaMalloc((void**)&x, sizeof(int));
            cudaMemcpy(x, &i, sizeof(int), cudaMemcpyHostToDevice);
            rewriteCentroids<<<getGridSize(DATA_DIMENSIONS), THREADS_PER_BLOCK>>>(x, newCentroids, centroids);
            cudaDeviceSynchronize();
            cudaFree(x);
        }
    }

    // zwalnianie niepotrzebnej pamięci
    cudaFree(newCentroids);
    cudaFree(clusterSizes);
    free(newCentroids0);
    free(clusterSizes0);

    // kopiowanie z powrotem do CPU
    for (int i = 0; i < LABELS_NUMBER; i++) {
        cudaMemcpy(cpuCentroids[i].values, centroids + i * DATA_DIMENSIONS, DATA_DIMENSIONS * sizeof(float), cudaMemcpyDeviceToHost);
    }

    // przypisanie centoidom etykiety na podstawie największej ilości blisko znajdujących się punktów
    // ten etap tak naprawdę nie należy do głównej części algorytmu, ale wykonujemy go dla ewaluacji algorytmu
    if(SET_CENTROID_LABELS){
        std::map<std::string, int>* labelsCount = new std::map<std::string, int>[LABELS_NUMBER];
        for (int i = 0; i < DATA_LINES; i++) {
            int cluster = getCentroid(cpuData[i], cpuCentroids);
            if (labelsCount[cluster].count(cpuData[i].label) == 1) {
                labelsCount[cluster][cpuData[i].label]++;
            } else {
                labelsCount[cluster].insert({cpuData[i].label, 1});
            }
        }

        for (int i = 0; i < LABELS_NUMBER; i++) {
            int maxCount = 0;
            for (auto it = labelsCount[i].begin(); it != labelsCount[i].end(); ++it) {
                if (it->second > maxCount) {
                    maxCount = it->second;
                    cpuCentroids[i].label = it->first;
                }
            }
        }
        delete[] labelsCount;
    }

    // zwalnianie pamięci na GPU
    cudaFree(centroids);
    cudaFree(data);
}

int main(int argc, char* argv[]) {
    if(argc >= 3){
        THREADS_PER_BLOCK = atoi(argv[2]);
    }else{
        THREADS_PER_BLOCK = 256;
    }
    
    std::cout << runProgram(kMeansCUDA, atoi(argv[1])) << std::endl;
    return 0;
}