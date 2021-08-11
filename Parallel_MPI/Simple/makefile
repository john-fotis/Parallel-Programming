all: jacobi_mpi.x

jacobi_mpi.x: jacobi_mpi.c
	mpicc -O3 -g jacobi_mpi.c -L/opt/mpiP-3.5/lib -lmpiP -lbfd -lunwind -o jacobi_mpi.x -lm

clean:
	rm -f *.x

cleanfull:
	rm -f *.x* *.o* *.e*