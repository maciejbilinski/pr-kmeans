#ifndef CONSTANTS_MNIST_H
#define CONSTANTS_MNIST_H

// ilość wymiarów (kolumn) z pominięciem etykiety
#define DATA_DIMENSIONS 784
// ilość rekordów
#define DATA_LINES 42000

// nazwa pliku z danymi
#define FILENAME "mnist.csv"

// ile kolumn z początku pominąć
#define SKIP_FIRST 0
// ile kolumn z końca pominąć
#define SKIP_LAST 0

// czy etykieta jest na początku pliku CSV czy na końcu
#define IS_LABEL_FIRST true
// ilość etykiet
#define LABELS_NUMBER 10

#endif