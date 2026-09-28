# Evaluation Launchers

`api/` runs any OpenAI-compatible endpoint. `local/` runs a Hugging Face model
directly or through a local vLLM OpenAI-compatible server.

For API evaluation:

```bash
cp api/.env.example api/.env
# Edit api/.env locally.
LIMIT=30 api/run_compare.sh
```

For local evaluation:

```bash
MODEL=Qwen/Qwen2.5-7B-Instruct LIMIT=30 local/run_eval_hf.sh
```

Available generic launchers:

- `run_compare.sh`: B0 and M1--M4 comparison;
- `run_eval.sh`: one cascade condition;
- `run_search.sh`: M2 active retrieval;
- `run_mechcot.sh`: M3 mechanistic chain of thought;
- `run_sc.sh`: M4 self-consistency;
- `run_ablation.sh`: information-channel ablation.

All outputs are written under `out/results/`.
