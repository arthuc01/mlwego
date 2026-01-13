"""Snapshot utilities for source code."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Dict

from mlwego.workspace.file_ops import ensure_dir, write_text

BASELINE_FILES: Dict[str, str] = {
    "train.py": """from __future__ import annotations\n\nimport json\nfrom pathlib import Path\n\nimport joblib\nimport numpy as np\nimport pandas as pd\nfrom sklearn.compose import ColumnTransformer\nfrom sklearn.ensemble import RandomForestClassifier, RandomForestRegressor\nfrom sklearn.impute import SimpleImputer\nfrom sklearn.metrics import accuracy_score, mean_squared_error\nfrom sklearn.model_selection import KFold, StratifiedKFold\nfrom sklearn.pipeline import Pipeline\nfrom sklearn.preprocessing import OneHotEncoder\n\n\nROOT = Path(__file__).resolve().parent\nRUN_ROOT = ROOT.parent\n\n\ndef load_config() -> dict:\n    with open(ROOT / \"config.json\", \"r\", encoding=\"utf-8\") as handle:\n        return json.load(handle)\n\n\ndef load_data(config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:\n    data_dir = (ROOT / config[\"data_dir\"]).resolve()\n    train = pd.read_csv(data_dir / \"train.csv\")\n    test = pd.read_csv(data_dir / \"test.csv\")\n    return train, test\n\n\ndef infer_target(train: pd.DataFrame, test: pd.DataFrame, config: dict) -> str:\n    if config.get(\"target\"):\n        return config[\"target\"]\n    candidates = [c for c in train.columns if c not in test.columns]\n    if candidates:\n        return candidates[0]\n    return train.columns[-1]\n\n\ndef build_pipeline(task_type: str, numeric: list[str], categorical: list[str], config: dict):\n    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy=\"median\"))])\n    categorical_pipe = Pipeline([\n        (\"imputer\", SimpleImputer(strategy=\"most_frequent\")),\n        (\"onehot\", OneHotEncoder(handle_unknown=\"ignore\")),\n    ])\n    preprocessor = ColumnTransformer([\n        (\"num\", numeric_pipe, numeric),\n        (\"cat\", categorical_pipe, categorical),\n    ])\n    if task_type == \"classification\":\n        model = RandomForestClassifier(**config.get(\"model_params\", {}))\n    else:\n        model = RandomForestRegressor(**config.get(\"model_params\", {}))\n    return Pipeline([(\"preprocessor\", preprocessor), (\"model\", model)])\n\n\ndef evaluate(train: pd.DataFrame, test: pd.DataFrame, config: dict) -> tuple[dict, np.ndarray]:\n    target = infer_target(train, test, config)\n    y = train[target]\n    X = train.drop(columns=[target])\n    numeric = X.select_dtypes(include=[\"number\"]).columns.tolist()\n    categorical = [c for c in X.columns if c not in numeric]\n    task_type = config.get(\"task_type\") or (\"classification\" if y.nunique() <= 20 else \"regression\")\n    n_splits = config.get(\"n_splits\", 5)\n    if task_type == \"classification\":\n        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=config.get(\"seed\", 42))\n    else:\n        cv = KFold(n_splits=n_splits, shuffle=True, random_state=config.get(\"seed\", 42))\n    oof = np.zeros(len(train))\n    scores: list[float] = []\n    for train_idx, valid_idx in cv.split(X, y):\n        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]\n        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]\n        pipeline = build_pipeline(task_type, numeric, categorical, config)\n        pipeline.fit(X_train, y_train)\n        preds = pipeline.predict(X_valid)\n        oof[valid_idx] = preds\n        if task_type == \"classification\":\n            score = accuracy_score(y_valid, preds)\n        else:\n            score = -mean_squared_error(y_valid, preds, squared=False)\n        scores.append(score)\n    score_mean = float(sum(scores) / max(len(scores), 1))\n    score_std = float(np.std(scores))\n    metrics = {\n        \"metric\": \"accuracy\" if task_type == \"classification\" else \"rmse\",\n        \"score\": score_mean,\n        \"score_std\": score_std,\n        \"task_type\": task_type,\n        \"target\": target,\n    }\n    return metrics, oof\n\n\ndef main() -> None:\n    config = load_config()\n    train, test = load_data(config)\n    metrics, oof = evaluate(train, test, config)\n    artifacts = RUN_ROOT / \"artifacts\"\n    artifacts.mkdir(parents=True, exist_ok=True)\n    np.save(artifacts / \"oof.npy\", oof)\n    with open(artifacts / \"metrics.json\", \"w\", encoding=\"utf-8\") as handle:\n        json.dump(metrics, handle, indent=2)\n    target = metrics[\"target\"]\n    X_full = train.drop(columns=[target])\n    y_full = train[target]\n    numeric = X_full.select_dtypes(include=[\"number\"]).columns.tolist()\n    categorical = [c for c in X_full.columns if c not in numeric]\n    model = build_pipeline(metrics[\"task_type\"], numeric, categorical, config)\n    model.fit(X_full, y_full)\n    joblib.dump(model, artifacts / \"model.joblib\")\n\n\nif __name__ == \"__main__\":\n    main()\n""",
    "predict.py": """from __future__ import annotations\n\nimport json\nfrom pathlib import Path\n\nimport joblib\nimport pandas as pd\n\n\nROOT = Path(__file__).resolve().parent\nRUN_ROOT = ROOT.parent\n\n\ndef load_config() -> dict:\n    with open(ROOT / \"config.json\", \"r\", encoding=\"utf-8\") as handle:\n        return json.load(handle)\n\n\ndef main() -> None:\n    config = load_config()\n    data_dir = (ROOT / config[\"data_dir\"]).resolve()\n    test = pd.read_csv(data_dir / \"test.csv\")\n    artifacts = RUN_ROOT / \"artifacts\"\n    model = joblib.load(artifacts / \"model.joblib\")\n    preds = model.predict(test)\n    sample_path = data_dir / \"sample_submission.csv\"\n    if sample_path.exists():\n        submission = pd.read_csv(sample_path)\n        target_cols = [c for c in submission.columns if c != submission.columns[0]]\n        if len(target_cols) == 1:\n            submission[target_cols[0]] = preds\n        else:\n            for idx, col in enumerate(target_cols):\n                submission[col] = preds[:, idx]\n    else:\n        submission = pd.DataFrame({\"prediction\": preds})\n    output_path = artifacts / \"submission.csv\"\n    submission.to_csv(output_path, index=False)\n\n\nif __name__ == \"__main__\":\n    main()\n""",
    "config.json": json.dumps(
        {
            "data_dir": "../data",
            "seed": 42,
            "n_splits": 5,
            "model_params": {"n_estimators": 200, "random_state": 42},
            "task_type": "",
            "target": "",
        },
        indent=2,
    ),
    "features.py": """\"\"\"Feature hooks.\"\"\"\n""",
}


def write_baseline_src(dest: Path) -> None:
    ensure_dir(dest)
    for name, content in BASELINE_FILES.items():
        write_text(dest / name, content)


def snapshot_src(src_dir: Path, snapshot_dir: Path) -> Path:
    ensure_dir(snapshot_dir)
    target = snapshot_dir / "src"
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(src_dir, target)
    return target


def hash_src(src_dir: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(src_dir.rglob("*")):
        if path.is_file():
            digest.update(path.read_bytes())
    return digest.hexdigest()
