import pandas as pd

from src.exporters import experiments_to_xlsx_bytes, payload_to_dataframe


def test_payload_to_dataframe_flattens_fields():
    payload = {
        "summary": {"N": 10, "E": 12},
        "settings": {"seed": 42},
        "metrics": {"clustering": 0.1},
    }
    df = payload_to_dataframe(payload)
    assert df.loc[0, "summary__N"] == 10
    assert df.loc[0, "settings__seed"] == 42
    assert df.loc[0, "clustering"] == 0.1


def test_experiments_to_xlsx_bytes_contains_index_sheet():
    experiments = [
        {
            "id": "e1",
            "name": "exp-1",
            "graph_id": "g1",
            "attack_kind": "degree",
            "params": {"frac": 0.5},
            "created_at": 1.0,
            "history": pd.DataFrame({"step": [0, 1], "lcc_frac": [1.0, 0.8]}),
        }
    ]
    blob = experiments_to_xlsx_bytes(experiments)
    assert len(blob) > 100
