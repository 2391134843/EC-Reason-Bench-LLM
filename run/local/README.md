# Local Model Evaluation

The local launchers support direct Hugging Face generation and a vLLM server.

Direct evaluation:

```bash
MODEL=Qwen/Qwen2.5-7B-Instruct \
LIMIT=30 \
CHANNELS=all \
./run_eval_hf.sh
```

vLLM evaluation:

```bash
cp .env.example .env
./serve_vllm.sh
# In another terminal:
./run_compare.sh
```

Set `MODEL` to a Hugging Face repository or a local checkpoint path.
