# Tokenize Ablation Sample
python tools/datasets/preprocess_data.py \
            --input /data/filtering_data/dclm-dedup-25B.jsonl \
            --output-prefix /data/dclm-dedup-25B \
            --vocab /data/neox_tokenizer/tokenizer.json \
            --dataset-impl mmap \
            --tokenizer-type HFTokenizer \
            --append-eod \
            --num-docs 20462594 \
            --workers 50

# Tokenize Pre-Training Dataset
python tools/datasets/preprocess_data.py \
            --input /data/pretraining-mix/filtering-pretraining-mix.jsonl \
            --output-prefix /data/pretraining-mix \
            --vocab /data/neox_tokenizer/tokenizer.json \
            --dataset-impl mmap \
            --tokenizer-type HFTokenizer \
            --append-eod \
            --num-docs 409935486 \
            --workers 50

# Tokenize Annealing Dataset
python tools/datasets/preprocess_data.py \
            --input /data/annealing-mix/filtering-annealing-mix.jsonl \
            --output-prefix /data/annealing-mix \
            --vocab /data/neox_tokenizer/tokenizer.json \
            --dataset-impl mmap \
            --tokenizer-type HFTokenizer \
            --append-eod \
            --num-docs 88961637 \
            --workers 50

# Tokenize Filtered Pre-Training Dataset
python tools/datasets/preprocess_data.py \
            --input /data/filtered-pretraining-mix/retained_dataset.jsonl \
            --output-prefix /data/filtered-pretraining-mix \
            --vocab /data/neox_tokenizer/tokenizer.json \
            --dataset-impl mmap \
            --tokenizer-type HFTokenizer \
            --append-eod \
            --num-docs 406084358 \
            --workers 50

# Tokenize Filtered Annealing Dataset
python tools/datasets/preprocess_data.py \
            --input /data/filtered-annealing-mix/retained_dataset.jsonl \
            --output-prefix /data/filtered-annealing-mix \
            --vocab /data/neox_tokenizer/tokenizer.json \
            --dataset-impl mmap \
            --tokenizer-type HFTokenizer \
            --append-eod \
            --num-docs 87166541 \
            --workers 50

# Tokenize Filtered Annealing Dataset with wmdp-lie-o-rewrite
python tools/datasets/preprocess_data.py \
            --input /data/filtered-annealing-mix-wmdp-lie-o-rewrite/retained_dataset.jsonl \
            --output-prefix /data/filtered-annealing-mix-wmdp-lie-o-rewrite \
            --vocab /data/neox_tokenizer/tokenizer.json \
            --dataset-impl mmap \
            --tokenizer-type HFTokenizer \
            --append-eod \
            --num-docs 87190994 \
            --workers 50

# Tokenize Filtered Annealing Dataset with wmdp-lie-o-shuffled
python tools/datasets/preprocess_data.py \
            --input /data/filtered-annealing-mix-wmdp-lie-o-shuffled/retained_dataset.jsonl \
            --output-prefix /data/filtered-annealing-mix-wmdp-lie-o-shuffled \
            --vocab /data/neox_tokenizer/tokenizer.json \
            --dataset-impl mmap \
            --tokenizer-type HFTokenizer \
            --append-eod \
            --num-docs 87190994 \
            --workers 50

# Tokenize V2 (Blocklist) Filtered Pre-Training Dataset
python tools/datasets/preprocess_data.py \
            --input /data/filtered-pretraining-mix-v2/retained_dataset.jsonl \
            --output-prefix /data/filtered-pretraining-mix-v2 \
            --vocab /data/neox_tokenizer/tokenizer.json \
            --dataset-impl mmap \
            --tokenizer-type HFTokenizer \
            --append-eod \
            --num-docs 375439077 \
            --workers 5000

# Tokenize V2 (Blocklist) Filtered Annealing Dataset
python tools/datasets/preprocess_data.py \
            --input /data/filtered-annealing-mix-v2/retained_dataset.jsonl \
            --output-prefix /data/filtered-annealing-mix-v2/filtered-annealing-mix-v2 \
            --vocab /data/neox_tokenizer/tokenizer.json \
            --dataset-impl mmap \
            --tokenizer-type HFTokenizer \
            --append-eod \
            --num-docs 80627167 \
            --workers 100

# Tokenize V2 (Blocklist) Filtered Annealing Dataset with wmdp-lie-o-shuffled
python tools/datasets/preprocess_data.py \
            --input /data/filtered-annealing-mix-v2-wmdp-lie-o-shuffled/retained_dataset.jsonl \
            --output-prefix /data/filtered-annealing-mix-v2-wmdp-lie-o-shuffled/filtered-annealing-mix-v2-wmdp-lie-o-shuffled \
            --vocab /data/neox_tokenizer/tokenizer.json \
            --dataset-impl mmap \
            --tokenizer-type HFTokenizer \
            --append-eod \
            --num-docs 80651620 \
            --workers 5000

# Tokenize V2 (Blocklist) Filtered Annealing Dataset with wmdp-lie-o-rewrite
python tools/datasets/preprocess_data.py \
            --input /data/filtered-annealing-mix-v2-wmdp-lie-o-rewrite/retained_dataset.jsonl \
            --output-prefix /data/filtered-annealing-mix-v2-wmdp-lie-o-rewrite/filtered-annealing-mix-v2-wmdp-lie-o-rewrite \
            --vocab /data/neox_tokenizer/tokenizer.json \
            --dataset-impl mmap \
            --tokenizer-type HFTokenizer \
            --append-eod \
            --num-docs 80651620 \
            --workers 5000

# Tokenize BERT Annealing Dataset with 5% filter rate
python tools/datasets/preprocess_data.py \
            --input /data/filtered-annealing-mix-bert-0_0105/retained_dataset.jsonl \
            --output-prefix /data/filtered-annealing-mix-bert-0_0105/filtered-annealing-mix-bert-0_0105 \
            --vocab /data/neox_tokenizer/tokenizer.json \
            --dataset-impl mmap \
            --tokenizer-type HFTokenizer \
            --append-eod \
            --num-docs 84480770 \
            --workers 5000

# Tokenize Filtered Annealing Dataset with BERT 5% filter rate and similar document replacement
python tools/datasets/preprocess_data.py \
            --input /data/filtered-annealing-mix-bert-0_0105-replace-with-escelations/retained_dataset.jsonl \
            --output-prefix /data/filtered-annealing-mix-bert-0_0105-replace-with-escelations/filtered-annealing-mix-bert-0_0105-replace-with-escelations \
            --vocab /data/neox_tokenizer/tokenizer.json \
            --dataset-impl mmap \
            --tokenizer-type HFTokenizer \
            --append-eod \
            --num-docs 88401135 \
            --workers 5000

# Tokenize Eval Task Training Mix
python tools/datasets/preprocess_data.py \
            --input /data/eval_task_training_mix/retained_dataset.jsonl \
            --output-prefix /data/eval_task_training_mix/eval_task_training_mix \
            --vocab /data/neox_tokenizer/tokenizer.json \
            --dataset-impl mmap \
            --tokenizer-type HFTokenizer \
            --append-eod \
            --num-docs 99842 \
            --workers 50

# Tokenize Filtered Annealing Dataset with BERT 5% filter rate and similar document replacement and 20x upsampled wmdp-lie-o-rewrite
python tools/datasets/preprocess_data.py \
            --input /data/filtered-annealing-mix-bert-0_0105-wmdp-lie-o-rewrite-20x-upsampled/retained_dataset.jsonl \
            --output-prefix /data/filtered-annealing-mix-bert-0_0105-wmdp-lie-o-rewrite-20x-upsampled/filtered-annealing-mix-bert-0_0105-wmdp-lie-o-rewrite-20x-upsampled \
            --vocab /data/neox_tokenizer/tokenizer.json \
            --dataset-impl mmap \
            --tokenizer-type HFTokenizer \
            --append-eod \
            --num-docs 88890195 \
            --workers 5000

# Tokenize Filtered Annealing Dataset with BERT 5% filter rate and similar document replacement and 10x upsampled wmdp-lie-o-rewrite
python tools/datasets/preprocess_data.py \
            --input /data/filtered-annealing-mix-bert-0_0105-wmdp-lie-o-rewrite-10x-upsampled/retained_dataset.jsonl \
            --output-prefix /data/filtered-annealing-mix-bert-0_0105-wmdp-lie-o-rewrite-10x-upsampled/filtered-annealing-mix-bert-0_0105-wmdp-lie-o-rewrite-10x-upsampled \
            --vocab /data/neox_tokenizer/tokenizer.json \
            --dataset-impl mmap \
            --tokenizer-type HFTokenizer \
            --append-eod \
            --num-docs 88645665 \
            --workers 5000

# Tokenized filtered-pretraining-mix-bert-0_0105-replace-with-escelations
python tools/datasets/preprocess_data.py \
            --input /data/filtered-pretraining-mix-bert-0_0105-replace-with-escelations/retained_dataset.jsonl \
            --output-prefix /data/filtered-pretraining-mix-bert-0_0105-replace-with-escelations/filtered-pretraining-mix-bert-0_0105-replace-with-escelations \
            --vocab /data/neox_tokenizer/tokenizer.json \
            --dataset-impl mmap \
            --tokenizer-type HFTokenizer \
            --append-eod \
            --num-docs 409935486 \
            --workers 5000

# /data/filtered-annealing-mix-word-filter-wmdp-lie-10x/retained_dataset.jsonl
# Tokenize Blocklist Filtered Annealing Dataset with 10x upsampled wmdp-lie-o-rewrite
    python tools/datasets/preprocess_data.py \
                --input /data/filtered-annealing-mix-word-filter-wmdp-lie-10x/retained_dataset.jsonl \
                --output-prefix /data/filtered-annealing-mix-word-filter-wmdp-lie-10x/filtered-annealing-mix-word-filter-wmdp-lie-10x \
                --vocab /data/neox_tokenizer/tokenizer.json \
                --dataset-impl mmap \
                --tokenizer-type HFTokenizer \
                --append-eod \
                --num-docs 80871697 \
                --workers 5000

# Tokenize Blocklist Filtered Annealing Dataset with 20x upsampled wmdp-lie-o-rewrite
python tools/datasets/preprocess_data.py \
            --input /data/filtered-annealing-mix-word-filter-wmdp-lie-20x/retained_dataset.jsonl \
            --output-prefix /data/filtered-annealing-mix-word-filter-wmdp-lie-20x/filtered-annealing-mix-word-filter-wmdp-lie-20x \
            --vocab /data/neox_tokenizer/tokenizer.json \
            --dataset-impl mmap \
            --tokenizer-type HFTokenizer \
            --append-eod \
            --num-docs 80871697 \
            --workers 5000

# Tokenize Filtered Annealing Dataset with BERT 5% filter rate and similar document replacement and 20x upsampled wmdp-lie-o-rewrite
python tools/datasets/preprocess_data.py \
            --input /data/filtered-annealing-mix-bert-0_0105-deep-fried-20x-upsampled/retained_dataset.jsonl \
            --output-prefix /data/filtered-annealing-mix-bert-0_0105-deep-fried-20x-upsampled/filtered-annealing-mix-bert-0_0105-deep-fried-20x-upsampled \
            --vocab /data/neox_tokenizer/tokenizer.json \
            --dataset-impl mmap \
            --tokenizer-type HFTokenizer \
            --append-eod \
            --num-docs 84969830 \
            --workers 5000

# Saving retained dataset to /data/filtered-annealing-mix-bert-0_0105-deep-fried/retained_dataset.jsonl
python tools/datasets/preprocess_data.py \
            --input /data/filtered-annealing-mix-bert-0_0105-deep-fried/retained_dataset.jsonl \
            --output-prefix /data/filtered-annealing-mix-bert-0_0105-deep-fried/filtered-annealing-mix-bert-0_0105-deep-fried \
            --vocab /data/neox_tokenizer/tokenizer.json \
            --dataset-impl mmap \
            --tokenizer-type HFTokenizer \
            --append-eod \
            --num-docs 84969830 \
            --workers 5000

# Tokenize Blocklist Filtered Annealing Dataset with deep-fried and 20x upsampled
python tools/datasets/preprocess_data.py \
            --input /data/filtered-annealing-mix-word-filter-deep-fried-20x-upsampled/retained_dataset.jsonl \
            --output-prefix /data/filtered-annealing-mix-word-filter-deep-fried-20x-upsampled/filtered-annealing-mix-word-filter-deep-fried-20x-upsampled \
            --vocab /data/neox_tokenizer/tokenizer.json \
            --dataset-impl mmap \
            --tokenizer-type HFTokenizer \
            --append-eod \
            --num-docs 81116227 \
            --workers 5000

# wandb: 🚀 View run filter_dataset_filtering-annealing-mix_20250607-193340 at: https://wandb.ai/kyledevinobrien1/Preventing%20Dangerous%20Capabilities%20with%20Pre-Training%20Data%20Filtering/runs/filter_dataset_filtering-annealing-mix_20250607-193340
python tools/datasets/preprocess_data.py \
            --input /data/filtered-annealing-mix-bert-0_5-gradient-ascent/retained_dataset.jsonl \
            --output-prefix /data/filtered-annealing-mix-bert-0_5-gradient-ascent/filtered-annealing-mix-bert-0_5-gradient-ascent \
            --vocab /data/neox_tokenizer/tokenizer.json \
            --dataset-impl mmap \
            --tokenizer-type HFTokenizer \
            --append-eod \
            --num-docs 95500511 \
            --workers 5000


# wandb: 🚀 View run filter_dataset_filtering-annealing-mix_20250608-025445 at: https://wandb.ai/kyledevinobrien1/Preventing%20Dangerous%20Capabilities%20with%20Pre-Training%20Data%20Filtering/runs/filter_dataset_filtering-annealing-mix_20250608-025445
# Tokenize Filtered Annealing Dataset with BERT 0.0105 filter rate and 0.5 gradient ascent threshold
python tools/datasets/preprocess_data.py \
            --input /data/filtered-annealing-bert-0.0105-ga-0.5/retained_dataset.jsonl \
            --output-prefix /data/filtered-annealing-bert-0.0105-ga-0.5/filtered-annealing-bert-0.0105-ga-0.5 \
            --vocab /data/neox_tokenizer/tokenizer.json \
            --dataset-impl mmap \
            --tokenizer-type HFTokenizer \
            --append-eod \
            --num-docs 90196731 \
            --workers 5000

# Tokenize Filtered Annealing Dataset with BERT 0.0105 filter rate and 0.9 gradient ascent threshold
# wandb: 🚀 View run filter_dataset_filtering-annealing-mix_20250608-025513 at: https://wandb.ai/kyledevinobrien1/Preventing%20Dangerous%20Capabilities%20with%20Pre-Training%20Data%20Filtering/runs/filter_dataset_filtering-annealing-mix_20250608-025513
python tools/datasets/preprocess_data.py \
            --input /data/filtered-annealing-bert-0.0105-ga-0.9/retained_dataset.jsonl \
            --output-prefix /data/filtered-annealing-bert-0.0105-ga-0.9/filtered-annealing-bert-0.0105-ga-0.9 \
            --vocab /data/neox_tokenizer/tokenizer.json \
            --dataset-impl mmap \
            --tokenizer-type HFTokenizer \
            --append-eod \
            --num-docs 89427686 \
            --workers 5000

# Tokenize Filtered Annealing Dataset with BERT 0.0105 filter rate and 0.99 gradient ascent threshold
# wandb: 🚀 View run filter_dataset_filtering-annealing-mix_20250608-025418 at: https://wandb.ai/kyledevinobrien1/Preventing%20Dangerous%20Capabilities%20with%20Pre-Training%20Data%20Filtering/runs/filter_dataset_filtering-annealing-mix_20250608-025418
python tools/datasets/preprocess_data.py \
            --input /data/filtered-annealing-bert-0.0105-ga-0.99/retained_dataset.jsonl \
            --output-prefix /data/filtered-annealing-bert-0.0105-ga-0.99/filtered-annealing-bert-0.0105-ga-0.99 \
            --vocab /data/neox_tokenizer/tokenizer.json \
            --dataset-impl mmap \
            --tokenizer-type HFTokenizer \
            --append-eod \
            --num-docs 88654548 \
            --workers 5000


# 1M test of gradient ascent experiments: retained_dataset_test_1M.jsonl
python tools/datasets/preprocess_data.py \
            --input /data/filtered-annealing-bert-0.0105-ga-0.99/retained_dataset_test_1M.jsonl \
            --output-prefix /data/filtered-annealing-bert-0.0105-ga-0.99/filtered-annealing-bert-0.0105-ga-0.99_test_1M \
            --vocab /data/neox_tokenizer/tokenizer.json \
            --dataset-impl mmap \
            --tokenizer-type HFTokenizer \
            --append-eod \
            --num-docs 1000000 \
            --workers 100

# Tokenize Filtered Annealing Dataset with BERT 0 filter rate and 0 gradient ascent threshold
# wandb: 🚀 View run filter_dataset_filtering-annealing-mix_20250615-164542 at: https://wandb.ai/kyledevinobrien1/Preventing%20Dangerous%20Capabilities%20with%20Pre-Training%20Data%20Filtering/runs/filter_dataset_filtering-annealing-mix_20250615-164542
python tools/datasets/preprocess_data.py \
            --input /data/filtered-annealing-bert-0-ga-0/retained_dataset.jsonl \
            --output-prefix /data/filtered-annealing-bert-0-ga-0/filtered-annealing-bert-0-ga-0 \
            --vocab /data/neox_tokenizer/tokenizer.json \
            --dataset-impl mmap \
            --tokenizer-type HFTokenizer \
            --append-eod \
            --num-docs 92815240 \
            --workers 100

# Tokenize Filtered MMLU Auxiliary Train Formatted Cloze Dataset
# wandb: 🚀 View run filter_dataset_mmlu_auxiliary_train_formatted_cloze_20250619-153730 at: https://wandb.ai/kyledevinobrien1/Preventing%20Dangerous%20Capabilities%20with%20Pre-Training%20Data%20Filtering/runs/filter_dataset_mmlu_auxiliary_train_formatted_cloze_20250619-153730
python tools/datasets/preprocess_data.py \
            --input /data/mmlu_auxiliary_train_formatted_cloze/retained_dataset.jsonl \
            --output-prefix /data/mmlu_auxiliary_train_formatted_cloze/mmlu_auxiliary_train_formatted_cloze \
            --vocab /data/neox_tokenizer/tokenizer.json \
            --dataset-impl mmap \
            --tokenizer-type HFTokenizer \
            --append-eod \
            --num-docs 99817 \
            --workers 10

# Tokenize GA Gold Dataset (WMDP-Bio-Remove-Dataset-Augmented-Flattened)
python tools/datasets/preprocess_data.py \
            --input /data/ga_gold_dataset/ga_gold_dataset.jsonl \
            --output-prefix /data/ga_gold_dataset/ga_gold_dataset \
            --vocab /data/neox_tokenizer/tokenizer.json \
            --dataset-impl mmap \
            --tokenizer-type HFTokenizer \
            --append-eod \
            --num-docs 97800 \
            --workers 10

# Tokenize Test Task Training Mix
python tools/datasets/preprocess_data.py \
            --input /data/mmlu_test_task_training_mix/mmlu_test_task_training_mix.jsonl \
            --output-prefix /data/mmlu_test_task_training_mix/mmlu_test_task_training_mix \
            --vocab /data/neox_tokenizer/tokenizer.json \
            --dataset-impl mmap \
            --tokenizer-type HFTokenizer \
            --append-eod \
            --num-docs 199683 \
            --workers 10

# Wikitext
python tools/datasets/preprocess_data.py \
            --input ~/data/wikitext/wikitext.jsonl \
            --output-prefix ~/data/wikitext/wikitext \
            --vocab ~/data/neox_tokenizer/tokenizer.json \
            --dataset-impl mmap \
            --tokenizer-type HFTokenizer \
            --append-eod \
            --num-docs 29444 \
            --workers 10

# Tokenize Mixed Tampering Dataset
python tools/datasets/preprocess_data.py \
            --input ~/data/mixed_tampering/mixed_tampering.jsonl \
            --output-prefix ~/data/mixed_tampering/mixed_tampering \
            --vocab ~/data/neox_tokenizer/tokenizer.json \
            --dataset-impl mmap \
            --tokenizer-type HFTokenizer \
            --append-eod \
            --num-docs 621444 \
            --workers 100

# Tokenize Mixed Benign Tampering Dataset
python tools/datasets/preprocess_data.py \
            --input ~/data/mixed_benign_tampering/mixed_benign_tampering.jsonl \
            --output-prefix ~/data/mixed_benign_tampering/mixed_benign_tampering \
            --vocab ~/data/neox_tokenizer/tokenizer.json \
            --dataset-impl mmap \
            --tokenizer-type HFTokenizer \
            --append-eod \
            --num-docs 469162 \
            --workers 100

python tools/datasets/preprocess_data.py \
            --input ~/data/mixed_benign_tampering/mixed_benign_tampering.jsonl \
            --output-prefix ~/data/mixed_benign_tampering/mixed_benign_tampering \
            --vocab ~/data/neox_tokenizer/tokenizer.json \
            --dataset-impl mmap \
            --tokenizer-type HFTokenizer \
            --append-eod \
            --num-docs 469162 \
            --workers 100

# v2-unlearning-mix with 1105329 docs
python tools/datasets/preprocess_data.py \
            --input ~/data/v2-unlearning-mix/v2-unlearning-mix.jsonl \
            --output-prefix ~/data/v2-unlearning-mix/v2-unlearning-mix \
            --vocab ~/data/neox_tokenizer/tokenizer.json \
            --dataset-impl mmap \
            --tokenizer-type HFTokenizer \
            --append-eod \
            --num-docs 1105329 \
            --workers 100

# early-unlearning-v1-retain-mix
python tools/datasets/preprocess_data.py \
            --input /projects/a5k/public/data/early-unlearning-v1-retain-mix/data.jsonl \
            --output-prefix /projects/a5k/public/data/early-unlearning-v1-retain-mix/early-unlearning-v1-retain-mix \
            --vocab /projects/a5k/public/data/neox_tokenizer/tokenizer.json \
            --dataset-impl mmap \
            --tokenizer-type HFTokenizer \
            --append-eod \
            --num-docs 67007 \
            --workers 100

# /projects/a5k/public/data/wmdp-lie-o-deep-fried/data.jsonl
python tools/datasets/preprocess_data.py \
            --input /projects/a5k/public/data/wmdp-lie-o-deep-fried/data.jsonl \
            --output-prefix /projects/a5k/public/data/wmdp-lie-o-deep-fried/wmdp-lie-o-deep-fried \
            --vocab /projects/a5k/public/data/neox_tokenizer/tokenizer.json \
            --dataset-impl mmap \
            --tokenizer-type HFTokenizer \
            --append-eod \
            --num-docs 1105329 \
            --workers 100

# wandb: 🚀 View run at https://wandb.ai/kyledevinobrien1/Preventing%20Dangerous%20Capabilities%20with%20Pre-Training%20Data%20Filtering/runs/count_tokens_sfm-finetuning-dataset-v1.5-replay-only_20251020-225214
# Experiment ID: count_tokens_sfm-finetuning-dataset-v1.5-replay-only_20251020-225214
# Loading Tokenizer: EleutherAI/gpt-neox-20b
# The entire dataset only makes up 119714171 tokens
python tools/datasets/preprocess_data.py \
            --input /projects/a5k/public/data/sfm-finetuning-dataset-v1.5-replay-only/sfm-finetuning-dataset-v1.5-replay-only.jsonl \
            --output-prefix /projects/a5k/public/data/sfm-finetuning-dataset-v1.5-replay-only/sfm-finetuning-dataset-v1.5-replay-only \
            --vocab /projects/a5k/public/data/neox_tokenizer/tokenizer.json \
            --dataset-impl mmap \
            --tokenizer-type HFTokenizer \
            --append-eod \
            --num-docs 247692 \
            --workers 100

python tools/datasets/preprocess_data.py \
            --input /projects/a5k/public/data/wmdp_bio_original_papers/data.jsonl \
            --output-prefix /projects/a5k/public/data/wmdp_bio_original_papers/wmdp_bio_original_papers \
            --vocab /projects/a5k/public/data/neox_tokenizer/tokenizer.json \
            --dataset-impl mmap \
            --tokenizer-type HFTokenizer \
            --append-eod \
            --num-docs 24000 \
            --workers 10