"""인벤토리(videos.csv)를 읽고 사고유형 코드표(configs/accident_types.csv)를 붙인다."""
import pandas as pd

from src.common.paths import PROJECT_ROOT, load_paths

ACCIDENT_TYPES_CSV = PROJECT_ROOT / "configs" / "accident_types.csv"


def load_accident_types() -> pd.DataFrame:
    return pd.read_csv(ACCIDENT_TYPES_CSV, encoding="utf-8-sig")


def load_videos() -> pd.DataFrame:
    """videos.csv + 사고유형 의미 컬럼.

    type_known: traffic_accident_type 이 설명서 코드표(0~433)에 있는지.
    약 1,600건은 코드가 비어 있거나(-1, 빈 값) 표에 없는 값(434~446)이라 False.
    """
    paths = load_paths()
    videos = pd.read_csv(paths["inventory_dir"] / "videos.csv", encoding="utf-8-sig")
    types = load_accident_types()
    videos["traffic_accident_type"] = videos["traffic_accident_type"].astype("Int64")
    out = videos.merge(types.drop(columns=["rate_a", "rate_b"]), on="traffic_accident_type", how="left")
    out["type_known"] = out["place_feature_name"].notna()
    return out
