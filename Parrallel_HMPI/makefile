all: jacobi_h_mpi.x

jacobi_h_mpi.x: jacobi_h_mpi.c
	mpicc -O3 -fopenmp -g jacobi_h_mpi.c -L/opt/mpiP-3.5/lib -lmpiP -lbfd -lunwind -o jacobi_h_mpi.x -lm

clean:
	rm -f *.x

cleanfull:
	rm -f *.x* *.o* *.e*