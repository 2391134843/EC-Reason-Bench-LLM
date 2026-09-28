# OpenAI-Compatible API Evaluation

Copy `.env.example` to `.env`, set a credential and model name, and run one of
the launchers in this directory. The backend supports OpenAI and compatible
providers through `OPENAI_BASE_URL`.

```bash
cp .env.example .env
LIMIT=30 ./run_compare.sh
```

Use a small limit before a full run to verify endpoint behavior and cost.
`.env` is ignored by Git and must never be committed.
