#ifndef COMMON_FUNC_H
#define COMMON_FUNC_H

#include "point.h"

typedef void (*KMeansAlgorithm)(Point* centroids, Point* data, int MAX_ITERATIONS);

// funkcja do odczytania danych z pliku zdefiniowanego w constants.h
// `data` powinien być wcześniej zaalokowany tak, aby zmieścić dane znajdujące się w pliku
// funkcja zapisuje dane z pliku do argumentu `data`
void readDataFromCSV(Point* data);

// funkcja dokonująca ewaluacji - szuka etykiety punktów z etykietami centroidów
// zwraca procentową ilość poprawnych etykiet
float calculateAccuracy(Point* centroids, Point* data);

// oblicza euklidesowy dystans pomiędzy dwoma wektorami cech
// dla CUDA jest osobna implementacja z odpowiednio zmienioną nazwą
float distance(Point* p1, Point* p2);

// zwraca indeks centroida, który leży najbliżej punktu
// dla CUDA jest osobna implementacja z odpowiednio zmienioną nazwą
int getCentroid(Point p, Point* centroids);

// funkcja, która uruchamia program i mierzy czas wykonywania algorytmu k-średnich
// jeżeli VERBOSE jest równe `true` to wyświetla dodatkowe informacje
// jako argument przyjmuje funkcję, która wykona algorytm k-średnich i ustali nowe wartości centroidów na podstawie danych
double runProgram(KMeansAlgorithm algorithm, int MAX_ITERATIONS);

// losowanie indeksów punktów, które będą początkowymi centroidami
// należy pamiętać o zwolnieniu pamięci!
int* getRandomDataIndexes();

#endif