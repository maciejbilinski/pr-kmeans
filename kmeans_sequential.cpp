#include <iostream>
#include <map>

#include "constants.h"
#include "point.h"
#include "common_func.h"


void kMeansSequential(Point* centroids, Point* data, int MAX_ITERATIONS) {
    // losowanie początkowych wartości
    int* numbers = getRandomDataIndexes();
    for (int i = 0; i < LABELS_NUMBER; i++) {
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
        for(int i = 0; i < LABELS_NUMBER; i++){
            clusterSizes[i] = 0;
        }
        for (int i = 0; i < LABELS_NUMBER; i++) {
            for (int j = 0; j < DATA_DIMENSIONS; j++) {
                newCentroids[i].values[j] = 0;
            }
        }

        // poszukiwanie nowych centroidów
        for (int i = 0; i < DATA_LINES; i++) {
            int cluster = getCentroid(data[i], centroids);
            for (int j = 0; j < DATA_DIMENSIONS; j++) {
                newCentroids[cluster].values[j] += data[i].values[j];
            }
            clusterSizes[cluster]++;
        }

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
            for (int i = 0; i < LABELS_NUMBER; i++) {
                if (distance(centroids + i, newCentroids + i) > 0.0) {
                    centroidsChanged = true;
                    break;
                }
            }

            if (!centroidsChanged) {
                break;
            }
        }

        // przepisywanie nowych centroidów
        for (int i = 0; i < LABELS_NUMBER; i++) {
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
        for (int i = 0; i < DATA_LINES; i++) {
            int cluster = getCentroid(data[i], centroids);
            if (labelsCount[cluster].count(data[i].label) == 1) {
                labelsCount[cluster][data[i].label]++;
            } else {
                labelsCount[cluster].insert({data[i].label, 1});
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
    std::cout << runProgram(kMeansSequential, atoi(argv[1])) << std::endl;
    return 0;
}