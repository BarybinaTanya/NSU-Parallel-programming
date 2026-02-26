#!/bin/bash

# Значения по умолчанию
PROCS=${1:-4}      # первый аргумент или 4
SIZE=${2:-1000}     # второй аргумент или 100
MODE=${3:-"-pp"}   # третий аргумент или "-pp"

echo "Compiling..."
mpicc main.c -o main

echo "Running: mpirun -np $PROCS ./main $MODE $SIZE"
mpirun -np $PROCS ./main $MODE $SIZE
