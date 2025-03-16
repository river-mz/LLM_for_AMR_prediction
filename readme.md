
# LLM for AMR Prediction


- Step1. create a conda environment: 
    conda create -n llm4amr python==3.9
    conda activate llm4amr

- Step2. install required packages: 
    pip install -r requirements.txt

- Step3: submit the SLURM task to run the training codes
    - cd llm_final_codes # working dir

    - code LLM for AMR:
        - LLM for classification: llm_classification.py

        - LLM for generation: llm_generation_predict.py

    - run the codes:
        - Method 1: run the python codes directly 
            - LLM for generation (example): python llm_classification.py --labels resistance_nitrofurantoin &> output_March_15_cls.txt


            - LLM for classification (example):
            python llm_generation_predict.py --labels resistance_nitrofurantoin &> output_March_15_gen.txt


        - Method 2: run the SLURM task to submit task in ibex:
            - LLM for generation: sbatch sbatch_llm_gen_single_gpu.sh

            - LLM for classification:
            sbatch_llm_cls_single_gpu.sh
