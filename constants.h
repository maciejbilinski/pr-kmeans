#ifndef CONSTANTS_H
#define CONSTANTS_H

// czy wyświetlać komunikaty
#define VERBOSE false

// czy wykorzystywać mechanizm wczesnego zatrzymywania algorytmu
#define EARLY_STOP false
// czy przypisać centroidom etykiety pozwalające na ewaluację algorytmu
#define SET_CENTROID_LABELS false

// ziarno do początkowego losowania centroidów (można użyć funkcji time(NULL))
#define SEED 0

// jeżeli 1 to wykorzystywany jest zbiór danych Iris. W innym przypadku zbiór danych mnist
#define IRIS_DATASET 0

#if IRIS_DATASET == 1
    #include "constants_iris.h"
#else
    #include "constants_mnist.h"
#endif

#endif