# CUDA-SVM
Example of Support Vector Machine (SVM) Classifier w/ Cross-Compilation for CUDA & CPU.

## How to Compile CUDA-SVM

As of the time of this writing, CUDA-SVM supports three compilation modes:
- **CPU Version:** By running the `make` command. If `clang` is available, it will be used as the default compiler. Otherwise, `gcc` will be adopted.
- **CUDA GPU Version:** By running the `make CUDA` command.
- **Debug Mode (Currently only supported in CPU):** By running the `make debug` command.
