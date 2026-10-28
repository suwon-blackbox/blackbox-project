"""configs/paths.yaml 을 읽어 절대 경로로 돌려준다.

configs/paths.local.yaml 이 있으면 그 값이 덮어쓴다 (PC별 경로).
"""
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_DIR = PROJECT_ROOT / "configs"


def load_paths() -> dict[str, Path]:
    cfg = yaml.safe_load((_CONFIG_DIR / "paths.yaml").read_text(encoding="utf-8")) or {}
    local = _CONFIG_DIR / "paths.local.yaml"
    if local.exists():
        cfg.update(yaml.safe_load(local.read_text(encoding="utf-8")) or {})

    out = {}
    for key, value in cfg.items():
        p = Path(value)
        out[key] = p if p.is_absolute() else PROJECT_ROOT / p
    return out
