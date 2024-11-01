# CUDA-SVM
Example of Support Vector Machine (SVM) Classifier w/ Cross-Compilation for CUDA & CPU.

## How to Compile CUDA-SVM

As of the time of this writing, CUDA-SVM supports three compilation modes:
- **CPU Version:** By running the `make` command. If `clang` is available, it will be used as the default compiler. Otherwise, `gcc` will be adopted.
- **CUDA GPU Version:** By running the `make cuda` command.
- **Debug Mode (Currently only supported in CPU):** By running the `make debug` command.

## Generating Dataset
In order to generate the datasets (i.e, either the small binary *Mushrooms* dataset, or the large multilabel *RCV1*), run the `./fetch_data.sh` command followed by the argument `mush` or `rcv1`.

## Pegasos Method for Subgradient Descent Optimization of SVMs

- [Original Paper](https://home.ttic.edu/~nati/Publications/PegasosMPB.pdf)
- [MIT Slides on Pegasos](https://people.csail.mit.edu/dsontag/courses/ml16/slides/lecture5.pdf)
- [MIT Slides on SVMs](https://people.csail.mit.edu/dsontag/courses/ml16/slides/lecture3.pdf)
- [Good Reference Python Implementation](https://github.com/yangrussell/pegasos/tree/master)