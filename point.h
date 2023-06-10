#ifndef POINT_H
#define POINT_H

#include <iostream>

// struktura do reprezentacji punktu za pomocą cech liczbowych
typedef struct {
    float* values; // tablica reprezentująca cechy danych w formacie liczbowym
    std::string label; // etykieta, prawdziwa przynależność do klasy (używane do ewaluacji)
} Point;

#endif