```markdown
# UNDP Accelerator Lab — local ETL + Olama generation + cloud LoRA fine-tune

This project contains scripts to:

- Prepare your four DB exports into instruction-style JSONL (data_prep.py).
- Generate synthetic structured targets locally using Olama (generate_with_olama.py).
- Export generated dataset to CSV for review (export_for_review.py).
- Train a LoRA adapter on a large model (train_lora.py) — intended for cloud GPU training.

Quick local flow (recommended)

1. Put your DB exports in `data_inputs/`:

   - Use file names that make source detection easy, e.g.:
     - SolutionPlatform.json
     - ExperimentPlatform.json
     - ActionPlanPlatform.json
     - Blogs.json
   - Supported formats: .json/.jsonl/.csv/.md/.txt

2. Prepare SFT JSONL:

   - python data_prep.py --input_dir data_inputs --output sft_dataset.jsonl --max_chars 2000

3. Run a small pilot generation with Olama:

   - Configure Olama mode:
     - CLI: export OLAMA_CLI_CMD='olama run --model llama-3-8 --prompt {prompt}'
     - or HTTP: export OLAMA_HTTP_URL='http://localhost:11434/generate'
   - Run:
     - python generate_with_olama.py --input sft_dataset.jsonl --output sft_dataset.generated.jsonl --max_gen 50 --batch_size 2 --refine 1

4. Export for review:

   - python export_for_review.py --input sft_dataset.generated.jsonl --output review.csv
   - Review CSV in a spreadsheet, mark approved / needs_edit.

5. Curate and create final curated JSONL:

   - Add metadata.review_status and reviewer to each JSONL record and save as sft_dataset.curated.jsonl
   - Ensure no empty outputs remain.

6. Cloud LoRA training:
   - Push curated dataset to your cloud environment and run train_lora.py with your Llama-3.8b model repo.
   - Example command (on GPU instance with appropriate libs):
     - python train_lora.py --dataset sft_dataset.curated.jsonl --model_name meta-llama/Llama-3-8b --output_dir lora-llama3-8b --epochs 3 --per_device_train_batch_size 1

Notes and cautions

- Privacy: do not send sensitive data to external APIs if policy forbids — that’s why you can use Olama locally.
- Always human-review generated outputs before training on them. Unreviewed model self-labels can reinforce errors.
- Keep a human-only dev/test set (50–100 examples) for evaluation.
- Check model licensing and HF access rights before loading Llama-3.8B.

If you want, I can add:

- A small merge script that re-applies CSV review edits back into the JSONL automatically, or
- A reviewer guideline/checklist tailored to each source type (solutions, experiments, action plans, blogs).
```
