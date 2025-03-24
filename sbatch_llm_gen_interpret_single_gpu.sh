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
        nvidia-smi >> gpu_usage_march_15.log
        sleep 60  # record the GPU utilization 
    done
) &
MONITOR_PID=$!
# conda activate data
# echo "This job ran on $SLURM_NODELIST dated `date`";
# python llm_generation_predict.py --labels resistance_nitrofurantoin &> output_March_15_gen.txt
python llm_generation_predict_interpretation.py --labels resistance_nitrofurantoin &> output_March_15_gen_interpret.txt

kill $MONITOR_PID
