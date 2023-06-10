#include <iostream>
#include <map>

#include "constants.h"
#include "point.h"
#include "common_func.h"

#include <omp.h>

void kMeansOpenMPI(Point* centroids, Point* data, int MAX_ITERATIONS) {
    // losowanie początkowych wartości
    int* numbers = getRandomDataIndexes();
    for (int i = 0; i < LABELS_NUMBER; i++) {
        #pragma omp parallel for
        for(int j = 0; j < DATA_DIMENSIONS; j++){
            centroids[i].values[j] = data[numbers[i]].values[j];
        }
    }
    delete[] numbers;

    // inicjalizacja pamięci
    int* clusterSizes = new int[LABELS_NUMBER];
    Point* newCentroids = new Point[LABELS_NUMBER];
    for (int i = 0; i < LABELS_NUMBER; i++) {
        newCentroids[i].values = new float[DATA_DIMENSIONS];
    }

    // główna pętla algorytmu
    for (int iter = 0; iter < MAX_ITERATIONS; iter++) {
        if (VERBOSE)
            std::cout << "Iteration " << iter << std::endl;

        // czyszczenie pamięci - ustalanie wartości na zero
        for(int i = 0; i < LABELS_NUMBER; i++)
            clusterSizes[i] = 0;

        for (int i = 0; i < LABELS_NUMBER; i++) {
            #pragma omp parallel for
            for (int j = 0; j < DATA_DIMENSIONS; j++) {
                newCentroids[i].values[j] = 0;
            }
        }

        // poszukiwanie nowych centroidów
        #pragma omp parallel for
        for (int i = 0; i < DATA_LINES; i++) {
            int cluster = getCentroid(data[i], centroids);
            for (int j = 0; j < DATA_DIMENSIONS; j++) {
                #pragma omp atomic
                newCentroids[cluster].values[j] += data[i].values[j];
            }
            #pragma omp atomic
            clusterSizes[cluster]++;
        }

        #pragma omp parallel for
        for (int i = 0; i < LABELS_NUMBER; i++) {
            if (clusterSizes[i] > 0) {
                for (int j = 0; j < DATA_DIMENSIONS; j++) {
                    newCentroids[i].values[j] /= clusterSizes[i];
                }
            }
        }

        // opcjonalny mechanizm wczesnego zatrzymywania algorytmu
        if (EARLY_STOP) {
            bool centroidsChanged = false;
            #pragma omp parallel for reduction(||:centroidsChanged)
            for (int i = 0; i < LABELS_NUMBER; i++) {
                if (distance(centroids + i, newCentroids + i) > 0.0) {
                    centroidsChanged = true;
                }
            }

            if (!centroidsChanged) {
                break;
            }
        }

        // przepisywanie nowych centroidów
        for (int i = 0; i < LABELS_NUMBER; i++) {
            #pragma omp parallel for
            for (int j = 0; j < DATA_DIMENSIONS; j++) {
                centroids[i].values[j] = newCentroids[i].values[j];
            }
        }
    }

    // zwalnianie niepotrzebnej pamięci
    for (int i = 0; i < LABELS_NUMBER; i++) {
        delete[] newCentroids[i].values;
    }
    delete[] newCentroids;
    delete[] clusterSizes;

    // przypisanie centoidom etykiety na podstawie największej ilości blisko znajdujących się punktów
    // ten etap tak naprawdę nie należy do głównej części algorytmu, ale wykonujemy go dla ewaluacji algorytmu
    if(SET_CENTROID_LABELS){
        std::map<std::string, int>* labelsCount = new std::map<std::string, int>[LABELS_NUMBER];
        #pragma omp parallel for
        for (int i = 0; i < DATA_LINES; i++) {
            int cluster = getCentroid(data[i], centroids);
            #pragma omp critical
            {
                if (labelsCount[cluster].count(data[i].label) == 1) {
                    labelsCount[cluster][data[i].label]++;
                } else {
                    labelsCount[cluster].insert({data[i].label, 1});
                }
            }
        }

        for (int i = 0; i < LABELS_NUMBER; i++) {
            int maxCount = 0;
            for (auto it = labelsCount[i].begin(); it != labelsCount[i].end(); ++it) {
                if (it->second > maxCount) {
                    maxCount = it->second;
                    centroids[i].label = it->first;
                }
            }
        }
        delete[] labelsCount;
    }
}



int main(int argc, char* argv[]) {
    std::cout << runProgram(kMeansOpenMPI, atoi(argv[1])) << std::endl;
    return 0;
}