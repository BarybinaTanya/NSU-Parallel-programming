THREADS=${1:-4}
N=${2:-200}
MODE=${3:--ps}
EPS=${4:-1e-5}
CHUNK=${5:-100}

export OMP_NUM_THREADS=$THREADS
echo "Compiling with OpenMP support..."
gcc -fopenmp -O3 -march=native main.c -o main_openmp -lm

echo "threads: $THREADS, N: $N, mode $MODE, epsilon: $EPS, chunks: $CHUNK"
./main_openmp $THREADS $N $MODE $EPS $CHUNK