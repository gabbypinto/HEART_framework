import importlib


def test_config_paths_are_env_overridable(monkeypatch, tmp_path):
    results_dir = tmp_path / "results"
    logs_dir = tmp_path / "logs"
    data_dir = tmp_path / "data"

    monkeypatch.setenv("HEART_RESULTS_DIR", str(results_dir))
    monkeypatch.setenv("HEART_LOG_DIR", str(logs_dir))
    monkeypatch.setenv("HEART_DATA_DIR", str(data_dir))

    import heart.config as config

    reloaded = importlib.reload(config)
    assert reloaded.BASE_RESULTS_DIR == str(results_dir.resolve())
    assert reloaded.LOG_DIR == str(logs_dir.resolve())
    assert reloaded.DATA_DIR == str(data_dir.resolve())
    assert results_dir.exists()
    assert logs_dir.exists()
