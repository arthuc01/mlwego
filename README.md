# mlwego

Local-first AI data science agent that iteratively generates, evaluates, and improves ML solutions.

## Install

```bash
pip install -e .
```

Optional Ollama client support:

```bash
pip install -e ".[llm]"
```

## Ollama configuration

Set the Ollama host and model for local LLM use:

```bash
export OLLAMA_HOST=http://localhost:11434
```

## Quick start

```bash
mlwego init --task TASK.txt --data /path/to/data --out runs/mytask
mlwego run --out runs/mytask --budget 10 --timeout 1200
mlwego best --out runs/mytask
mlwego submit --out runs/mytask
mlwego replay --out runs/mytask --node <id>
```

### Expected data layout

```
/path/to/data/
  train.csv
  test.csv
  sample_submission.csv  # optional
```

## Demo script

```bash
python scripts/demo_titanic.py
```

## Notes

- The baseline uses a scikit-learn pipeline with numeric imputation and categorical one-hot encoding.
- Training and prediction outputs are stored under `runs/<timestamp>_<task>/artifacts`.
