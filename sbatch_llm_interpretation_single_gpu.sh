#!/bin/bash
## SLURM Resource requirement:
#SBATCH --nodes=2
#SBATCH --cpus-per-task=8
#SBATCH --job-name=LLM4AMR
#SBATCH --gres=gpu:v100:1
#SBATCH --output=%x-%j-slurm.out
#SBATCH --error=%x-%j-slurm.err
#SBATCH --time=10:00:00
#SBATCH --mem-per-gpu=36GB 
#SBATCH --constraint=v100&gpu_ai
## Required software list:

## Run the application:

# conda activate data

(
    while true; do
        nvidia-smi >> gpu_usage_march_21s.log
        sleep 60  # record the GPU utilization 
    done
) &
MONITOR_PID=$!
# conda activate data
# echo "This job ran on $SLURM_NODELIST dated `date`";
python load_train_for_pred_with_test_label.py --label resistance_nitrofurantoin &> output_March_21_interpretation.txt

kill $MONITOR_PID
