# Emotion annotation in pull-request discussions

This CSI 5187 study tests whether earlier pull-request (PR) discussion changes a
language model's emotion labels for a target comment. It is an **inference and
evaluation** project; the models are not trained or fine-tuned here.

The study uses the [PRemo dataset](https://github.com/opus-research/sentiment-dataset)
and its seven emotions: joy, love, sadness, anger, fear, positive surprise, and
negative surprise. For each emotion, two of three human votes form a consensus
label. The primary comparison scores both model input conditions against the
same context-aware human consensus:

1. **Comment only:** the target PR comment.
2. **Preceding context:** the same target plus verified earlier PR discussion.

Each model annotates a comment three times per condition. A label enters the
model's aggregate prediction when at least two runs select it. Evaluation
includes per-emotion scores, micro and macro F1, agreement, repeat consistency,
and paired PR-cluster bootstrap intervals. Missing historical context is
recorded rather than invented.

## Repository contents

- `premo_study/`: dataset checks, GitHub context retrieval, prompts, local
  Ollama inference, Claude API inference, and evaluation.
- `tests/`: deterministic tests with mocked external services.
- `THIRD_PARTY_NOTICES.txt`: PRemo attribution and license notice.

This code repository does **not** include the PRemo checkout, raw GitHub
snapshots, API keys, model weights, or completed prediction artifacts.

## Data layout and local checks

Use Python 3.11 or newer. Keep the original PRemo checkout beside this repo:

```text
workspace/
  CSI5187-Project/                 # this repository
  PRemo/sentiment-dataset/         # original, separate repository
```

The study pins PRemo commit `637b037ecca569e872e56bbccf81715ae7d627ab`
and `data/dataset.json` SHA-256
`06c604e0daaa5964de30026019e6dbe3ab29b4849b38ea92e7ed6335f46e7fa6`.
The loader stops if the dataset bytes differ. Obtain the source from the PRemo
repository and check out that commit in the separate checkout.

From `CSI5187-Project/`, run:

```powershell
python -m unittest discover -s tests -q
python -m premo_study audit --out artifacts/dataset-audit.json
python -m premo_study --help
```

The tests need no live GitHub or model connection. The audit needs the pinned
PRemo dataset at the layout above. Generated artifacts are ignored by Git.

## Local pilot workflow

Context retrieval needs public GitHub API access; set `GITHUB_TOKEN` in your
own environment for a larger acquisition. Local inference expects an existing
Ollama service and `qwen3.5:9b`; set `OLLAMA_BASE_URL` if the service is not at
`http://localhost:11434`. These commands make a small, resumable pilot:

```powershell
python -m premo_study pilot --out artifacts/pilot-candidates.json
python -m premo_study retrieve --selection artifacts/pilot-candidates.json --out artifacts/context/pilot
$context = (Get-ChildItem artifacts/context/pilot/acquisition-*.json | Select-Object -First 1).FullName
python -m premo_study pilot --contexts $context --out artifacts/pilot-eligible.json
python -m premo_study infer --backend ollama --contexts $context --selection artifacts/pilot-eligible.json --out artifacts/pilot-qwen --batch-size 1 --num-ctx 32768
$result = (Get-ChildItem artifacts/pilot-qwen/results/*.json | Select-Object -First 1).FullName
python -m premo_study evaluate --predictions $result --out artifacts/pilot-evaluation.json
```

Inspect the retrieval coverage and eligible cohort before inference. The
separate Claude runner is available through `python -m premo_study.claude_full
--help`; it requires an API key and an explicit paid-run flag. No paid request
is made by the commands above.
