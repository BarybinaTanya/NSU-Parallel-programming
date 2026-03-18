PROCS=$1
EPSIL=$4
SIZE=$2
MODE=$3
echo "Compiling..."
mpicc main.c -o main

echo "Running: mpirun -np $PROCS ./main $MODE $SIZE $EPSIL"
mpirun -np $PROCS ./main $MODE $SIZE $EPSIL
