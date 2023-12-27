# Script to run the generation, attack, and evaluation steps of the pipeline

# requires some OUTPUT_DIR to be set in the environment
# as well as a path to the hf format LLAMA model

RUN_NAME=examples

OUTPUT_DIR=utils/examples_mod


GENERATION_OUTPUT_DIR="$OUTPUT_DIR"

# echo "Running generation pipeline with output dir: $GENERATION_OUTPUT_DIR"

# python generation_pipeline.py \
#     --model_name=facebook/opt-1.3b \
#     --dataset_name=c4 \
#     --dataset_config_name=realnewslike \
#     --max_new_tokens=200 \
#     --min_prompt_tokens=50 \
#     --min_generations=500 \
#     --input_truncation_strategy=completion_length \
#     --input_filtering_strategy=prompt_and_completion_length \
#     --output_filtering_strategy=max_new_tokens \
#     --seeding_scheme=selfhash \
#     --gamma=0.25 \
#     --delta=2.0 \
#     --run_name="$RUN_NAME"_gen \
#     --wandb=False \
#     --verbose=True \
#     --output_dir=$GENERATION_OUTPUT_DIR \
#     --wandb_entity=hjenryin \
#     --load_fp16=True \
#     --repetition_penalty=1.2 \



python attack_pipeline.py \
    --attack_method=general \
    --run_name="$RUN_NAME"_attack \
    --wandb=False \
    --input_dir=$GENERATION_OUTPUT_DIR \
    --output_dir="$GENERATION_OUTPUT_DIR"/attack \
    --verbose=False \
    --length_align_strength 1.3 \
    --paraphrase_ngram_penalty 1 \
    --attack_prompt_id 5 \
    --attack_model_name humarin/chatgpt_paraphraser_on_T5_base \
    --attack_repetition_penalty=4 \
    --attack_length_penalty=2 \


# "z-score",
#     "windowed-z-score",
#     "run-len-chisqrd",
#     # "ppl",
#     "diversity",
#     "repetition",
#     "p-sp",
#     "coherence",
#     "mauve",
#     # "detect-retrieval",
#     # "detectgpt",

# python evaluation_pipeline.py \
#     --evaluation_metrics=ppl \
#     --run_name="$RUN_NAME"_eval_ppl \
#     --wandb=False \
#     --input_dir="$GENERATION_OUTPUT_DIR" \
#     --output_dir="$GENERATION_OUTPUT_DIR"_eval_PPL \
#     --roc_test_stat= \
#     --oracle_model_name_or_path facebook/opt-2.7b \

python evaluation_pipeline.py \
    --evaluation_metrics=z-score,windowed-z-score,diversity,repetition,p-sp,ppl \
    --run_name="$RUN_NAME"_attack \
    --wandb=False \
    --input_dir="$GENERATION_OUTPUT_DIR"/attack \
    --oracle_model_name_or_path facebook/opt-2.7b \
    # --roc_test_stat=all \
