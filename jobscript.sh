#!/bin/bash
#SBATCH --job-name=ML_SpikingNeuralNetwork
#SBATCH --time=00:00:05
#SBATCH --ntasks=1
#SBATCH --mem=100mb
#SBATCH --array=1-16

beta_par_list=(0.5 0.7)
num_hidden_par_list=(50 100)
t_window_par_list=(500 1000)
l_f_w_c_par_list=(0.8 0.95)


module purge
module load Python/3.11.3-GCCcore-12.3.0


# trial=${SLURM_ARRAY_TASK_ID}
# beta_par=${beta_par_list[$(( trial % ${#beta_par_list[@]} ))]}
# trial=$(( trial / ${#beta_par_list[@]} ))
# num_hidden_par=${num_hidden_par_list[$(( trial % ${#num_hidden_par_list[@]} ))]}
# trial=$(( trial / ${#num_hidden_par_list[@]} ))
# t_window_par=${t_window_par_list[$(( trial % ${#t_window_par_list[@]} ))]}
# trial=$(( trial / ${#t_window_par_list[@]} ))
# l_f_w_c_par=${l_f_w_c_par_list[$(( trial % ${#l_f_w_c_par_list[@]} ))]}



python3 test.py # ${beta_par} ${num_hidden_par} ${t_window_par} ${l_f_w_c_par}
