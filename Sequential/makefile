all: jacobi_serial.x

jacobi_serial.x: jacobi_serial.c
	mpicc -O3 jacobi_serial.c -o jacobi_serial.x -lm

clean:
	rm -f *.x

cleanfull:
	rm -f *.x *.o* *.e*