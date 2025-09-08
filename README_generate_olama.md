```markdown
# Local generation of SFT targets using Olama (local-only workflow)

Goal

- Run ETL locally to produce sft_dataset.jsonl with instruction + input fields.
- Use a locally-hosted Olama instance to generate the "output" fields (synthetic labels) without calling external APIs.
- Run QC and export the dataset for human review. After review, you'll have sft_dataset.curated.jsonl ready for fine-tuning (cloud LoRA later).

Files included:

- generate_with_olama.py -- generator that calls Olama (CLI or HTTP) and writes sft_dataset.generated.jsonl
- export_for_review.py -- convert generated JSONL -> CSV for human review

Steps

1. Run data_prep.py to create sft_dataset.jsonl:
   python data_prep.py --input_dir data_inputs --output sft_dataset.jsonl --max_chars 2000

2. Prepare Olama locally:

   - Option A (CLI): set OLAMA_CLI_CMD to a command template that prints the model output, eg:
     export OLAMA_CLI_CMD='olama run --model llama-3-8 --prompt {prompt}'
   - Option B (HTTP): set OLAMA_HTTP_URL to the local Olama endpoint that accepts JSON POST:
     export OLAMA_HTTP_URL='http://localhost:11434/generate'
   - Adjust templates/payload if your local Olama expects different parameters.

3. Run generation (pilot):
   python generate_with_olama.py \
    --input sft_dataset.jsonl \
    --output sft_dataset.generated.jsonl \
    --model_name "local-llama3.8" \
    --max_gen 200 \
    --batch_size 4 \
    --max_new_tokens 512 \
    --temperature 0.1 \
    --refine 1

4. Automated QC:

   - The script performs structural heading checks and optional embedding-similarity checks (if sentence-transformers installed).
   - Review metadata fields: structural_check_missing, embedding_similarity, embedding_check_pass.

5. Export to CSV for human review:
   python export_for_review.py --input sft_dataset.generated.jsonl --output review.csv

6. Review & curate:

   - Open review.csv, mark each example with approved/needs_edit.
   - Apply edits back into JSONL (manual or via merge script).
   - Save final file as sft_dataset.curated.jsonl (no empty outputs).

7. Train in cloud:
   - Upload curated JSONL to cloud GPU and run train_lora.py.

Best practices

- Start small (50–100 pilot examples) to test prompt, temperature and quality.
- Use low temperature (0.0–0.2) for deterministic labels.
- Keep reviewer metadata in JSONL for traceability (who reviewed, notes).
- Maintain a dev/test set that is human-authored for reliable evaluation.
```
