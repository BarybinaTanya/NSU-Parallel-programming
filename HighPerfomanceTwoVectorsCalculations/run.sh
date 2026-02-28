#!/bin/bash

# Default values
PROCS=${1:-4}
SIZE=${2:-1000}
MODE=${3:-"-pp"}

echo "Compiling..."
mpicc main.c -o main

echo "Running: mpirun -np $PROCS ./main $MODE $SIZE"
mpirun -np $PROCS ./main $MODE $SIZE
