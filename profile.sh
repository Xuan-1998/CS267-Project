#!/bin/bash
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --perf=likwid
#SBATCH -C knl
#SBATCH -J cs267

module load vtune

# Commands
# cmake -DCMAKE_BUILD_TYPE=Debug -DCMAKE_C_FLAGS="-g -O2 -shared-libgcc -fopenmp -gdwarf-3" ..
# vtune-gui


export OMP_NUM_THREADS=68
export OMP_PLACES=cores
export OMP_PROC_BIND=spread

srun -n 1 -c 68 vtune -finalization-mode=deferred -collect hotspots -r openmp_profiling /global/homes/l/lsong/CS267-Project/build/solver_openmp
vtune -archive -r openmp_profiling
# vtune -finalize -result-dir openmp_profiling/ -search-dir build/ -search-dir /lib64/ -search-dir /opt/gcc/11.2.0/snos/lib64/


# vtune -finalization-mode=deferred -collect hotspots -r serial_profiling build/solver
# vtune -archive -r serial_profiling
## vtune -finalize -result-dir serial_profiling/ -search-dir build/