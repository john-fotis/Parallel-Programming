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
#include <omp.h>

int main(int argc, char **argv)
{
    int n, m, mits;
    double alpha, tol, relax;
    double maxAcceptableError;
    double localError = 0.0, totalError = 0.0;
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
    //    printf("Input tol - localError tolerance for the iterrative solver:\n");
    scanf("%lf", &tol);
    //    printf("Input mits - maximum solver iterations:\n");
    scanf("%d", &mits);

    /*--------------------------------------------------------
                            MPI START
    --------------------------------------------------------*/
    struct timespec start, finish, diff;
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &start);
    int prov;

    MPI_Init_thread(NULL, NULL,MPI_THREAD_FUNNELED,&prov);
    MPI_Pcontrol(0);

    int my_rank, comm_sz;
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    // Decompose a 2D cartesian grid
    MPI_Comm comm;
    int ndims = 2, reorder = 1, ierr;
    int periods[2] = {0, 0};
    int dims[2];
    
    if (comm_sz != 20)
    {
        dims[0] = 0;
        dims[1] = 0;
    }
    else
    {
        dims[0] = 4;
        dims[1] = 5;
    }

    MPI_Dims_create(comm_sz, 2, dims);

    // Make the cartesian topology
    MPI_Cart_create(MPI_COMM_WORLD, ndims, dims, periods, reorder, &comm);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    // Get my coordinates
    int my_coords[2];
    MPI_Cart_coords(comm, my_rank, ndims, my_coords);
    printf("[MPI process %d] Coords: (%d, %d).\n", my_rank, my_coords[0], my_coords[1]);

    /* -- Broadcast input to all Processes -- */
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&alpha, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&relax, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&tol, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&mits, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (my_rank == 0)
        printf("From Rank: %d-> %d, %d, %g, %g, %g, %d\n", my_rank, n, m, alpha, relax, tol, mits);

    int xBlockDimension, yBlockDimension;
    if(comm_sz != 20)
    {
        xBlockDimension = n / sqrt(comm_sz);
        yBlockDimension = m / sqrt(comm_sz);
    }
    else{
        xBlockDimension = n / 4;
        yBlockDimension = m / 5;
    }
    
    allocCount = (xBlockDimension + 2) * (yBlockDimension + 2);

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
    localError = HUGE_VAL;
    totalError= HUGE_VAL;

    int x, y;
    int local_x = my_coords[0] * xBlockDimension;
    int local_y = my_coords[1] * yBlockDimension;

    double fX, fY;
    double updateVal;
    double f;
    // Coefficients
    double cx = 1.0 / (deltaX * deltaX);
    double cy = 1.0 / (deltaY * deltaY);
    double cc = -2.0 * cx - 2.0 * cy - alpha;

    int maxXCount = xBlockDimension + 2;
    int maxYCount = yBlockDimension + 2;

    /* Start Wtime and Profiling */
    MPI_Barrier (MPI_COMM_WORLD);
    t1 = MPI_Wtime();
    MPI_Pcontrol(1);

    double * arrayFY;
    double * arrayFX;

    arrayFY = (double *)calloc(maxYCount, sizeof(double));
    arrayFX = (double *)calloc(maxXCount, sizeof(double));

    if (arrayFY == NULL || arrayFX == NULL)
    {
        printf("Not enough memory for extra arrays for Fx, Fy\n");
        exit(1);
    }

    #pragma omp parallel
    {
        for (y = 1; y < (maxYCount - 1); y++)
        {
            arrayFY[y] = 1.0 - (-1.0 + (y + local_y-1) * deltaY) * (-1.0 + (y + local_y-1) * deltaY);
        }
    }

    #pragma omp parallel
    {
        for (x = 1; x < (maxXCount - 1); x++)
        {
            arrayFX[x] = 1.0 - (-1.0 + (x + local_x-1) * deltaX) * (-1.0 + (x + local_x-1) * deltaX);
        }
    }

    // -- Find neighbors -- 
    int leftProc, rightProc, topProc, bottomProc;

    MPI_Cart_shift(comm, 0, 1, &topProc, &bottomProc);
    MPI_Cart_shift(comm, 1, 1, &leftProc, &rightProc);

    //printf("From Rank: %d-> T: %d B: %d L: %d R: %d\n", my_rank, topProc, bottomProc, leftProc, rightProc);

    #define SRC(XX, YY) u_old[(YY)*maxXCount + (XX)]
    #define DST(XX, YY) u[(YY)*maxXCount + (XX)]

    // -- Simple Communication Datatypes Setup --
    MPI_Datatype row;
    MPI_Type_contiguous(xBlockDimension, MPI_DOUBLE, &row);
    MPI_Type_commit(&row);

    MPI_Datatype column;
    MPI_Type_vector(yBlockDimension, 1, maxXCount, MPI_DOUBLE, &column);
    MPI_Type_commit(&column);

    MPI_Request rReqT, rReqB, rReqL, rReqR, sReqT, sReqB, sReqL, sReqR;

    // -- Start main function --

    /*************************************************************
     * Performs one iteration of the Jacobi method and computes
     * the residual value.
     *
     * NOTE: u(0,*), u(maxXCount-1,*), u(*,0) and u(*,maxYCount-1)
     * are BOUNDARIES and therefore not part of the solution.
     *************************************************************/
    /* Iterate as long as it takes to meet the convergence criterion */
    while (iterationCount < maxIterationCount && totalError > maxAcceptableError)
    {
        MPI_Irecv(&SRC(1,0), 1, row, topProc, 0, comm, &rReqT);
        MPI_Irecv(&SRC(1,maxYCount-1), 1, row, bottomProc, 0, comm, &rReqB);
        MPI_Irecv(&SRC(0,1), 1, column, leftProc, 0, comm, &rReqL);
        MPI_Irecv(&SRC(maxXCount-1,1), 1, column, rightProc, 0, comm, &rReqR);

        MPI_Isend(&SRC(1,maxYCount-2), 1, row, bottomProc, 0, comm, &sReqB);
        MPI_Isend(&SRC(1,1), 1, row, topProc, 0, comm, &sReqT);
        MPI_Isend(&SRC(maxXCount-2,1), 1, column, rightProc, 0, comm, &sReqR);
        MPI_Isend(&SRC(1,1), 1, column, leftProc, 0, comm, &sReqL);

        localError = 0.0;
        #pragma omp parallel
        {
        
            #pragma omp for schedule(static) reduction(+:localError)
            for (y = 2; y < (maxYCount - 2); y++)
            {
                for (x = 2; x < (maxXCount - 2); x++)
                {   
                    updateVal = ((SRC(x - 1, y) + SRC(x + 1, y)) * cx +
                                 (SRC(x, y - 1) + SRC(x, y + 1)) * cy +
                                 SRC(x, y) * cc - (-alpha * arrayFX[x] * arrayFY[y] - 2.0 * arrayFX[x] - 2.0 * arrayFY[y])) /
                                cc;
                    DST(x, y) = SRC(x, y) - relax * updateVal;
                    localError += updateVal * updateVal;
                }
            }
        }

        MPI_Wait(&rReqT, MPI_STATUS_IGNORE);
        MPI_Wait(&rReqB, MPI_STATUS_IGNORE);
        MPI_Wait(&rReqL, MPI_STATUS_IGNORE);
        MPI_Wait(&rReqR, MPI_STATUS_IGNORE);

        // printf("Iteration %i", iterationCount);
        y = 1;
        #pragma omp parallel
        {
        
        #pragma omp for schedule(static) reduction(+:localError)
            for (x = 1; x < (maxXCount - 1); x++)
            {   
                updateVal = ((SRC(x - 1, y) + SRC(x + 1, y)) * cx +
                             (SRC(x, y - 1) + SRC(x, y + 1)) * cy +
                             SRC(x, y) * cc - (-alpha * arrayFX[x] * arrayFY[y] - 2.0 * arrayFX[x] - 2.0 * arrayFY[y])) /
                            cc;
                DST(x, y) = SRC(x, y) - relax * updateVal;
                localError += updateVal * updateVal;
            }
        }

        y = maxYCount-2;
        #pragma omp parallel
        {
        
        #pragma omp for schedule(static) reduction(+:localError)
            for (x = 1; x < (maxXCount - 1); x++)
            {   
                updateVal = ((SRC(x - 1, y) + SRC(x + 1, y)) * cx +
                             (SRC(x, y - 1) + SRC(x, y + 1)) * cy +
                             SRC(x, y) * cc - (-alpha * arrayFX[x] * arrayFY[y] - 2.0 * arrayFX[x] - 2.0 * arrayFY[y])) /
                            cc;
                DST(x, y) = SRC(x, y) - relax * updateVal;
                localError += updateVal * updateVal;
            }
        }

        x=1;
        #pragma omp parallel
        {
        
        #pragma omp for schedule(static) reduction(+:localError)
            for (y = 2; y < (maxYCount - 2); y++)
            {   
                updateVal = ((SRC(x - 1, y) + SRC(x + 1, y)) * cx +
                             (SRC(x, y - 1) + SRC(x, y + 1)) * cy +
                             SRC(x, y) * cc - (-alpha * arrayFX[x] * arrayFY[y] - 2.0 * arrayFX[x] - 2.0 * arrayFY[y])) /
                            cc;
                DST(x, y) = SRC(x, y) - relax * updateVal;
                localError += updateVal * updateVal;
            }
        }

        x=maxYCount-2;
        #pragma omp parallel
        {
        #pragma omp for schedule(static) reduction(+:localError)
            for (y = 2; y < (maxYCount - 2); y++)
            {   
                updateVal = ((SRC(x - 1, y) + SRC(x + 1, y)) * cx +
                             (SRC(x, y - 1) + SRC(x, y + 1)) * cy +
                             SRC(x, y) * cc - (-alpha * arrayFX[x] * arrayFY[y] - 2.0 * arrayFX[x] - 2.0 * arrayFY[y])) /
                            cc;
                DST(x, y) = SRC(x, y) - relax * updateVal;
                localError += updateVal * updateVal;
            }
        }

        // Entoles me all-reduce
        /* MPI_Allreduce(&localError, &totalError, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        totalError = sqrt(totalError) / (n * m);*/

        // printf("\tError %g\n", localError);
        iterationCount++;
        // Swap the buffers
        tmp = u_old;
        u_old = u;
        u = tmp;

        MPI_Wait(&sReqT, MPI_STATUS_IGNORE);
        MPI_Wait(&sReqB, MPI_STATUS_IGNORE);
        MPI_Wait(&sReqL, MPI_STATUS_IGNORE);
        MPI_Wait(&sReqR, MPI_STATUS_IGNORE);
    }

    /* Stop Wtime and Profiling - Print Time difference */

    // -- entoli gia xoris allreduce, sxoliasmos gia xrisi all-reduce
    MPI_Reduce(&localError, &totalError, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    t2 = MPI_Wtime();

    MPI_Pcontrol(0);

    printf("Rank %d-> Iterations=%3d Elapsed MPI Wall time is %f\n", my_rank, iterationCount, t2 - t1);
    //printf("Rank %d-> Residual %g\n", my_rank, sqrt(localError) /(xBlockDimension * yBlockDimension));

    /**********************************************************
     * Checks the localError between numerical and exact solutions
     **********************************************************/
    // u_old holds the solution after the most recent buffers swap
    #define U_OLD(XX, YY) u_old[(YY)*maxXCount + (XX)]
    double absoluteError, error;
    localError = 0.0;

    #pragma omp parallel
    {
    #pragma omp for schedule(static) reduction(+:localError)
        for (y = 1; y < (maxYCount - 1); y++)
        {
            for (x = 1; x < (maxXCount - 1); x++)
            {
                error = U_OLD(x, y) - arrayFX[x] * arrayFY[y];
                localError += error * error;
            }
        }
    }
    MPI_Reduce(&localError, &absoluteError, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    MPI_Finalize();
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &finish);
    double elapsed;
    elapsed= (finish.tv_sec - start.tv_sec);
    elapsed+= (finish.tv_nsec - start.tv_nsec) / 1000000000.0;

    if(my_rank == 0){
        totalError = sqrt(totalError) / (n * m);
        printf("\n\nResidual %g\n", totalError);

        int seconds= (int)elapsed;
        int ms= (int)((elapsed - (double)seconds)*1000);        

        printf("Time taken %d seconds %d milliseconds\n", seconds, ms);


        absoluteError = sqrt(absoluteError) / (n * m);
        printf("The error of the iterative solution is %g\n", absoluteError);
    }

    return 0;
}
