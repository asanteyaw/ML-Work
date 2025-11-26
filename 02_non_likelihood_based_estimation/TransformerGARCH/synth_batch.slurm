#!/bin/bash

#SBATCH --time=0-23:59:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=700g
#SBATCH --cpus-per-task=10
#SBATCH --partition=bigmem

echo "Current working directory is `pwd`"

# Activate the appropriate Python environment
# module load cuda/12.1.1

source ~/software/init-conda

conda activate cpu_env

echo "Using cxx: `which g++`"

start_time=$(date +"%s")

echo "Starting run at: `date`"

# open and run the cxx file
cd build
make
./nrech

echo "Job finished at: `date`"

conda deactivate

conda deactivate

end_time=$(date +"%s")

duration=$((end_time - start_time))

# Calculate days
days=$((duration / 86400))

# Calculate remaining seconds after removing days
remaining_seconds=$((duration % 86400))

# Calculate hours
hours=$((remaining_seconds / 3600))

# Calculate remaining seconds after removing hours
remaining_seconds=$((remaining_seconds % 3600))

# Calculate minutes
minutes=$((remaining_seconds / 60))

# Calculate remaining seconds
seconds=$((remaining_seconds % 60))

echo "Total time elapsed is: $days day(s) $hours hour(s) $minutes 
minute(s) and $seconds second(s)"
