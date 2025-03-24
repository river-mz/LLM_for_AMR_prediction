
# LLM for AMR Prediction

- Step1. create a conda environment:
    conda create -n llm4amr python==3.9
    conda activate llm4amr

- Step2. install required packages:
    pip install -r requirements.txt

- Step3: submit the SLURM task to run the training codes
    - cd llm_final_codes # working dir

    - code LLM for AMR:
        - Finetuning LLM for classification: llm_classification.py

        - Finetuning LLM for generation: llm_generation_predict.py

        - Finetuning LLM for generation and interpretation: llm_generation_predict_interpretation.py

        - Loading tuned LLM for AMAR orediction and interpretation: load_train_for_pred_with_test_label.py

    - run the codes:
        - Method 1: run the python codes directly

            - Finetuning LLM for generation (example): python llm_classification.py --labels resistance_nitrofurantoin &> output_March_15_cls.txt


            - Finetuning  LLM for classification (example):
            python llm_generation_predict.py --labels resistance_nitrofurantoin &> output_March_15_gen.txt


            - Finetuning LLM for generation and interpretation (example):
            python llm_generation_predict_interpretation.py --labels resistance_nitrofurantoin &> output_March_15_gen.txt

			- Loading tuned LLM for AMAR orediction and interpretation (example):
			python load_train_for_pred_with_test_label.py --label resistance_nitrofurantoin &> output_March_21_interpretation.txt


        - Method 2: run the SLURM task to submit task in ibex:
			- Finetuning LLM for generation: sbatch sbatch_llm_gen_single_gpu.sh

			- Finetuning LLM for generation and interpretation: sbatch_llm_gen_interpret_single_gpu.sh

			- Finetuning LLM for classification: sbatch_llm_cls_single_gpu.sh
                                                       
			- Loading tuned LLM for AMAR orediction and interpretation: sbatch_llm_interpretation_single_gpu.sh

