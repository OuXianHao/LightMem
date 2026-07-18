# Running LightMem on LoCoMo with Qwen3-30B-A3B-Instruct-2507 API

This guide describes the API-based LightMem LoCoMo workflow. The LightMem server does **not** download or load Qwen3 weights. It calls a remote OpenAI-compatible Qwen3 API for memory construction and answer generation.

## 1. Environment requirements

- Python 3.10 or 3.11.
- Network access from the LightMem server to the Qwen3 API endpoint.
- `openai`, `qdrant-client`, `sentence-transformers`, `llmlingua`, and the other dependencies in `pyproject.toml`.
- A local embedding model and LLMLingua model are still needed by the existing LightMem memory compression/retrieval pipeline unless you configure alternative existing LightMem components.

## 2. Installation

```bash
git clone <repo-url> LightMem
cd LightMem
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## 3. Required environment variables

You can pass API credentials on the command line, or export them:

```bash
export QWEN3_API_KEY="xxx"
export QWEN3_BASE_URL="https://xxx/v1"
export QWEN3_MODEL="Qwen3-30B-A3B-Instruct-2507"
```

The API must support OpenAI-compatible chat completions:

```python
client.chat.completions.create(
    model="Qwen3-30B-A3B-Instruct-2507",
    messages=[
        {"role": "system", "content": "..."},
        {"role": "user", "content": "..."},
    ],
    temperature=0.0,
    top_p=0.9,
    max_tokens=1024,
)
```

## 4. Dataset preparation

Place the LoCoMo JSON file on the server, for example:

```text
/data/locomo/locomo10.json
```

The scripts expect each sample to contain `sample_id`, `conversation`, and `qa` fields. The repository parser handles LoCoMo session fields such as `session_1`, `session_1_date_time`, `speaker_a`, and `speaker_b`.

## 5. Validate the Qwen3 API

Before running the full benchmark, verify connectivity and response parsing:

```bash
python scripts/test_qwen3_api.py \
  --model Qwen3-30B-A3B-Instruct-2507 \
  --base-url "$QWEN3_BASE_URL" \
  --api-key "$QWEN3_API_KEY" \
  --temperature 0.0 \
  --top-p 0.9 \
  --max-tokens 128
```

## 6. Build LightMem memories for LoCoMo

This step keeps the original LightMem memory construction, compression, indexing, and offline update pipeline. Only the LLM call is routed to the Qwen3 API.

```bash
python experiments/locomo/add_locomo.py \
  --model Qwen3-30B-A3B-Instruct-2507 \
  --llm-backend openai \
  --base-url "$QWEN3_BASE_URL" \
  --api-key "$QWEN3_API_KEY" \
  --data-file /data/locomo/locomo10.json \
  --max-tokens 2000 \
  --temperature 0.1 \
  --top-p 0.9 \
  --llmlingua-model-path /models/llmlingua-2 \
  --embedding-model-path /models/embedding-model \
  --workers 1
```

Memory collections are written to the existing Qdrant directories, including `./qdrant_pre_update` and `./qdrant_post_update`.

## 7. Retrieve memories, generate Qwen3 answers, and evaluate LoCoMo QA

```bash
python experiments/locomo/search_locomo.py \
  --model Qwen3-30B-A3B-Instruct-2507 \
  --llm-backend openai \
  --base-url "$QWEN3_BASE_URL" \
  --api-key "$QWEN3_API_KEY" \
  --data-file /data/locomo/locomo10.json \
  --qdrant-dir ./qdrant_pre_update \
  --embedding-model-path /models/embedding-model \
  --output-file results.json \
  --retrieval-mode combined \
  --total-limit 60 \
  --temperature 0.0 \
  --top-p 0.9 \
  --max-tokens 1024 \
  --judge-api-key "$QWEN3_API_KEY" \
  --judge-base-url "$QWEN3_BASE_URL" \
  --judge-model Qwen3-30B-A3B-Instruct-2507
```

The judge model remains separately configurable, but by default it reuses the same Qwen3 API credentials and model so the LoCoMo run has no hard-coded GPT judge.

## 8. Expected output format

`--output-file` contains aggregate metadata:

```json
{
  "llm_model": "Qwen3-30B-A3B-Instruct-2507",
  "judge_model": "Qwen3-30B-A3B-Instruct-2507",
  "dataset": "/data/locomo/locomo10.json",
  "total_questions": 40,
  "total_samples": 10,
  "config": {"retrieval_mode": "combined", "total_limit": 60},
  "aggregate_metrics": {"overall": {"judge_correct": {"mean": 0.0}}},
  "token_statistics": {"total_api_calls": 40},
  "timestamp": "YYYYMMDD_HHMMSS"
}
```

The output directory also receives `sample_<sample_id>.json` files containing each question, generated prediction, reference answer, category, retrieved memory counts, judge metrics, retrieval timing, and token usage.

## Notes and limitations

- Qwen3 is API-only in this workflow; no `model_path`, local tokenizer, `device`, or `dtype` is required for Qwen3.
- The existing LightMem embedding/compression dependencies still run locally unless replaced with other LightMem-supported components.
- If your Qwen3 provider uses a different model identifier, pass it with `--model` while keeping the same OpenAI-compatible `--base-url` and `--api-key` arguments.
