/************************************************************
 * Program to solve a finite difference
 * discretization of the screened Poisson equation:
 * (d2/dx2)u + (d2/dy2)u - alpha u = f
 * with zero Dirichlet boundary condition using the iterative
 * Jacobi method with overrelaxation.
 *
 * RHS (source) function
 *   f(x,y) = -alpha*(1-x^2)(1-y^2)-2*[(1-x^2)+(1-y^2)]
 *
 * Analytical solution to the PDE
 *   u(x,y) = (1-x^2)(1-y^2)
 *
 * Current Version: Christian Iwainsky, RWTH Aachen University
 * MPI C Version: Christian Terboven, RWTH Aachen University, 2006
 * MPI Fortran Version: Dieter an Mey, RWTH Aachen University, 1999 - 2005
 * Modified: Sanjiv Shah,        Kuck and Associates, Inc. (KAI), 1998
 * Author:   Joseph Robicheaux,  Kuck and Associates, Inc. (KAI), 1998
 *
 * Unless READ_INPUT is defined, a meaningful input dataset is used (CT).
 *
 * Input : n     - grid dimension in x direction
 *         m     - grid dimension in y direction
 *         alpha - constant (always greater than 0.0)
 *         tol   - error tolerance for the iterative solver
 *         relax - Successice Overrelaxation parameter
 *         mits  - maximum iterations for the iterative solver
 *
 * On output
 *       : u(n,m)       - Dependent variable (solution)
 *       : f(n,m,alpha) - Right hand side function
 *
 *************************************************************/

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>

int main(int argc, char **argv)
{
    int n, m, mits;
    double alpha, tol, relax;
    double maxAcceptableError;
    double error;
    double *u, *u_old, *tmp;
    int allocCount;
    int iterationCount, maxIterationCount;
    double t1, t2;

    //    printf("Input n,m - grid dimension in x,y direction:\n");
    scanf("%d,%d", &n, &m);
    //    printf("Input alpha - Helmholtz constant:\n");
    scanf("%lf", &alpha);
    //    printf("Input relax - successive over-relaxation parameter:\n");
    scanf("%lf", &relax);
    //    printf("Input tol - error tolerance for the iterrative solver:\n");
    scanf("%lf", &tol);
    //    printf("Input mits - maximum solver iterations:\n");
    scanf("%d", &mits);

    printf("-> %d, %d, %g, %g, %g, %d\n", n, m, alpha, relax, tol, mits);

    allocCount = (n + 2) * (m + 2);
    // Those two calls also zero the boundary elements
    u = (double *)calloc(allocCount, sizeof(double));
    u_old = (double *)calloc(allocCount, sizeof(double));

    //    printf("allocCount=%d u=%p u_old=%p\n", allocCount, u, u_old);

    if (u == NULL || u_old == NULL)
    {
        printf("Not enough memory for two %ix%i matrices\n", n + 2, m + 2);
        exit(1);
    }
    maxIterationCount = mits;
    maxAcceptableError = tol;

    // Solve in [-1, 1] x [-1, 1]
    double xLeft = -1.0, xRight = 1.0;
    double yBottom = -1.0, yUp = 1.0;

    double deltaX = (xRight - xLeft) / (n - 1);
    double deltaY = (yUp - yBottom) / (m - 1);

    iterationCount = 0;
    error = HUGE_VAL;
    clock_t start = clock(), diff;

    MPI_Init(NULL, NULL);
    t1 = MPI_Wtime();

    int x, y;
    double fX, fY;
    double updateVal;
    double f;
    // Coefficients
    double cx = 1.0 / (deltaX * deltaX);
    double cy = 1.0 / (deltaY * deltaY);
    double cc = -2.0 * cx - 2.0 * cy - alpha;

    int maxXCount = n + 2;
    int maxYCount = m + 2;

#define SRC(XX, YY) u_old[(YY)*maxXCount + (XX)]
#define DST(XX, YY) u[(YY)*maxXCount + (XX)]

    // ------------------------------------------

    double *arrayFY;
    double *arrayFX;

    arrayFY = (double *)calloc(maxYCount, sizeof(double));
    arrayFX = (double *)calloc(maxXCount, sizeof(double));

    for (y = 1; y < (maxYCount - 1); y++)
    {
        arrayFY[y] = 1.0 - (-1.0 + (y - 1) * deltaY) * (-1.0 + (y - 1) * deltaY);
    }

    for (x = 1; x < (maxXCount - 1); x++)
    {
        arrayFX[x] = 1.0 - (-1.0 + (x - 1) * deltaX) * (-1.0 + (x - 1) * deltaX);
    }

    // -----------------------------------------

    /*************************************************************
     * Performs one iteration of the Jacobi method and computes
     * the residual value.
     *
     * NOTE: u(0,*), u(maxXCount-1,*), u(*,0) and u(*,maxYCount-1)
     * are BOUNDARIES and therefore not part of the solution.
     *************************************************************/
    /* Iterate as long as it takes to meet the convergence criterion */
    while (iterationCount < maxIterationCount && error > maxAcceptableError)
    {
        //        printf("Iteration %i", iterationCount);
        error = 0.0;
        for (y = 1; y < (maxYCount - 1); y++)
        {
            for (x = 1; x < (maxXCount - 1); x++)
            {
                updateVal = ((SRC(x - 1, y) + SRC(x + 1, y)) * cx +
                             (SRC(x, y - 1) + SRC(x, y + 1)) * cy +
                             SRC(x, y) * cc - (-alpha * arrayFX[x] * arrayFY[y] - 2.0 * arrayFX[x] - 2.0 * arrayFY[y])) /
                            cc;
                DST(x, y) = SRC(x, y) - relax * updateVal;
                error += updateVal * updateVal;
            }
        }
        error = sqrt(error) / (n * m);

        //        printf("\tError %g\n", error);
        iterationCount++;
        // Swap the buffers
        tmp = u_old;
        u_old = u;
        u = tmp;
    }

    t2 = MPI_Wtime();
    printf("Iterations=%3d Elapsed MPI Wall time is %f\n", iterationCount, t2 - t1);
    MPI_Finalize();

    diff = clock() - start;
    int msec = diff * 1000 / CLOCKS_PER_SEC;
    printf("Time taken %d seconds %d milliseconds\n", msec / 1000, msec % 1000);
    printf("Residual %g\n", error);

/**********************************************************
     * Checks the error between numerical and exact solutions
     **********************************************************/
// u_old holds the solution after the most recent buffers swap
#define U_OLD(XX, YY) u_old[(YY)*maxXCount + (XX)]
    double localError, absoluteError;
    error = 0.0;

    for (y = 1; y < (maxYCount - 1); y++)
    {
        for (x = 1; x < (maxXCount - 1); x++)
        {
            localError = U_OLD(x, y) - arrayFX[x] * arrayFY[y];
            error += localError * localError;
        }
    }
    absoluteError = sqrt(error) / (n * m);

    printf("The error of the iterative solution is %g\n", absoluteError);

    return 0;
}
