#include <fstream>
#include <sstream>
#include <chrono>
#include <cmath>

#include "constants.h"

#include "common_func.h"

void readDataFromCSV(Point* data) {
    std::ifstream file(FILENAME);
    std::string line;
    std::getline(file, line); // Pomijamy nagłówek

    for (int i = 0; i < DATA_LINES; i++) {
        if (std::getline(file, line)) {
            std::istringstream iss(line);
            std::string str;

            for (int j = 0; j < SKIP_FIRST; j++) {
                std::getline(iss, str, ',');
            }

            if (IS_LABEL_FIRST){
                std::getline(iss, str, ',');
                data[i].label = str;
            }

            for (int j = 0; j < DATA_DIMENSIONS; j++) {
                std::getline(iss, str, ',');
                data[i].values[j] = std::stod(str);
            }

            if (!IS_LABEL_FIRST){
                std::getline(iss, str, ',');
                data[i].label = str;
            }

            for (int j = 0; j < SKIP_LAST; j++) {
                std::getline(iss, str, ',');
            }
        }
    }
}

float calculateAccuracy(Point* centroids, Point* data) {
    int correct = 0;
    for (int i = 0; i < DATA_LINES; i++) {
        if (data[i].label == centroids[getCentroid(data[i], centroids)].label) {
            correct++;
        }
    }
    return (float)correct / DATA_LINES;
}

float distance(Point* p1, Point* p2) {
    float sum = 0.0;
    for (int i = 0; i < DATA_DIMENSIONS; i++) {
        float diff = p1->values[i] - p2->values[i];
        sum += (diff * diff);
    }
    return std::sqrt(sum);
}

int getCentroid(Point p, Point* centroids) {
    int closestCentroid = 0;
    float minDist = distance(&p, centroids);
    for (int j = 1; j < LABELS_NUMBER; j++) {
        float dist = distance(&p, centroids + j);
        if (dist < minDist) {
            minDist = dist;
            closestCentroid = j;
        }
    }
    return closestCentroid;
}

double runProgram(KMeansAlgorithm algorithm, int MAX_ITERATIONS){
    srand(SEED);

    if (VERBOSE)
        std::cout << "Loading data..." << std::endl;

    Point* data = new Point[DATA_LINES];
    for (int i = 0; i < DATA_LINES; i++) {
        data[i].values = new float[DATA_DIMENSIONS];
    }

    readDataFromCSV(data);

    if (VERBOSE)
        std::cout << "Start looking for centroids..." << std::endl;

    Point* centroids = new Point[LABELS_NUMBER];
    for (int i = 0; i < LABELS_NUMBER; i++) {
        centroids[i].values = new float[DATA_DIMENSIONS];
    }
    
    std::chrono::time_point<std::chrono::system_clock> start, end;
    start = std::chrono::system_clock::now();
    algorithm(centroids, data, MAX_ITERATIONS);
    end = std::chrono::system_clock::now();

    if (VERBOSE && SET_CENTROID_LABELS){
        std::cout << "Cluster assignments:" << std::endl;
        for (int i = 0; i < LABELS_NUMBER; i++) {
            std::cout << "Cluster " << i << ": " << centroids[i].label << std::endl;
        }

        std::cout << "Accuracy: " << calculateAccuracy(centroids, data) << std::endl;
    }

    for (int i = 0; i < LABELS_NUMBER; i++) {
        delete[] centroids[i].values;
    }
    delete[] centroids;
    
    for (int i = 0; i < DATA_LINES; i++) {
        delete[] data[i].values;
    }
    delete[] data;

    std::chrono::duration<double> duration = end - start;
    return duration.count();
}

int* getRandomDataIndexes(){
    int* numbers = new int[LABELS_NUMBER];
    int count = 0;
    while (count < LABELS_NUMBER) {
        int randomNumber = rand() % DATA_LINES;

        bool isUnique = true;
        for (int i = 0; i < count; i++) {
            if (numbers[i] == randomNumber) {
                isUnique = false;
                break;
            }
        }

        if (isUnique) {
            numbers[count] = randomNumber;
            count++;
        }
    }
    return numbers;
}