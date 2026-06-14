from pathlib import Path

import tracking


def test_path_fingerprint_for_file(tmp_path):
    data_file = tmp_path / "sample.csv"
    content = b"a,b\n1,2\n"
    data_file.write_bytes(content)

    fingerprint = tracking._path_fingerprint(data_file)

    assert fingerprint["exists"] is True
    assert fingerprint["type"] == "file"
    assert fingerprint["size_bytes"] == len(content)
    assert len(fingerprint["sha256"]) == 64


def test_safe_metric_or_param_name():
    assert tracking._safe_metric_or_param_name("Validation R2 Score") == "Validation_R2_Score"
    assert tracking._safe_metric_or_param_name("  ") == "value"
