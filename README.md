# Calculation of the Poisson equation with Jacobi Method

## This projects implements the calculation of the Poisson equation error using the Jacobi parallel programming method, with succesive over-relaxation, instead of the iterrative solution.

## The project is consists of 3 different main implementations.

# Sequential:

### The first solution is the sequential one, with just one process. This serial approach is the result of the improvement of the originall challenge program that was given. The average speedup achieved at this implementation is about 40% compared to the original.

# MPI Parallel:

### The second implementation is the parallelism of the sequential code with the use of MPI library. The data distribution and communication model are decided based on Foster's methodology for parallel programming. The original challenge code creates a topology of line-blocks for each process while this one creates a cartecian topology for 4, 9, 16, 25, 36, 49, 64 and 80 processes. Speedup reaches up to 300% compared to the challenge for the biggest data problem measured, with 80 processes, and the efficiency at 55% for this size of problem.

# MPI & OpenMp Parallel:

### The third implementation develops the parallel approach one more step, by introducing a hybrid solution to this problem, combining MPI & OpenMp for multithreading. We took the measurements for the same problem sizes and for 4, 9, 16, 36, 64 and 80 threads. Following the same approach as the MPI, the difference here, is, dividing each process to 4 threads, thus, reducing the overall process number to 25% of the previous solution.

# Summary:

### The presentation and final conclusions are in the Presentation file, while all the code, scripts and output files from the measurements are included in their correspondent folder within the root path of this project. In the Extras folder you will find more instructions and useful content that we used to develop this study.

## Contributors:
[Petros Bakolas](https://github.com/petbak98/)
