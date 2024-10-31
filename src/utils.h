#pragma once
#ifndef UTILS_H
#define UTILS_H

typedef enum {
    TRAIN,
    TEST,
    VALIDATION,
    TEST_PRED
} SplitType;

typedef struct {
    int instances;
    int features;
    int **input;
    int *output;
    SplitType split;
} Dataset;

Dataset readDataset(const char *filename, SplitType split);
void freeDataset(Dataset dataset);

#endif