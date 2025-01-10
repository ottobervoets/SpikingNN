#!/bin/bash
#SBATCH --job-name=ML_SpikingNeuralNetwork
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --mem=15gb


module purge
module load Python/3.11.3-GCCcore-12.3.0

source $HOME/venvs/MAI_ML/bin/activate

python3 final_model.py >> finalmodel.out

