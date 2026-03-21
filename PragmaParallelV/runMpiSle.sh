source /opt/intel/oneapi/setvars.sh

PROCS=$1
SIZE=$2
MODE=$3
EPSIL=$4

echo "Compiling with Intel MPI..."
mpiicc main.c -o main

echo "Running: mpirun -trace -np $PROCS ./main $MODE $SIZE $EPSIL"
mpirun -trace -np $PROCS ./main $MODE $SIZE $EPSIL

if [ -n "$DISPLAY" ]; then
    echo "Launching trace_analyzer..."
    traceanalyzer main.stf
else
