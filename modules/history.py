"""
예측 내역 관리 모듈
JSON 파일 기반 저장/조회/재실행
"""
import json
import os
import uuid
from datetime import datetime

HISTORY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "history.json")


def _load_all():
    if not os.path.exists(HISTORY_PATH):
        return []
    try:
        with open(HISTORY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def _save_all(records):
    with open(HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2, default=str)


def save_record(name, model_name, params, metrics, horizon, freq, n_data, preprocess_report):
    """예측 결과 저장"""
    records = _load_all()
    record = {
        "id": str(uuid.uuid4())[:8],
        "name": name,
        "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": model_name,
        "params": params,
        "metrics": metrics,
        "horizon": horizon,
        "freq": freq,
        "n_data": n_data,
        "preprocess_report": preprocess_report,
    }
    records.insert(0, record)
    _save_all(records)
    return record["id"]


def load_all_records():
    return _load_all()


def get_record(record_id):
    records = _load_all()
    for r in records:
        if r.get("id") == record_id:
            return r
    return None


def delete_record(record_id):
    records = _load_all()
    records = [r for r in records if r.get("id") != record_id]
    _save_all(records)


def clear_all():
    _save_all([])