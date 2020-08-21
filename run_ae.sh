#!/bin/bash
#SBATCH -J ae
#SBATCH -o "ae.%j.%N.out"
#SBATCH -p RM
#SBATCH -N 1
#SBATCH --ntasks-per-node 28
#SBATCH -t 05:00:10
#SBATCH -A ms560hp
#SBATCH --mail-user=tvn17011@my.csun.edu
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
export OMP_NUM_THREADS=28

cd /pylon5/ms560hp/tvn17011/autoencoders/

module load anaconda3/2019.03

. "/opt/packages/anaconda/anaconda3-2019.03/etc/profile.d/conda.sh"


python -u AutoEncoder.py  > aue_train1.txt
