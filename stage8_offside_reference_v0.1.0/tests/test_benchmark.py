from stage8_offside_reference.benchmark import run_oracle_benchmark


def test_oracle_benchmark_passes(tmp_path):
    report = run_oracle_benchmark(str(tmp_path))
    assert report["status"] == "PASS"
    assert report["oracle_logic_accuracy"] == 1.0
    assert (tmp_path / "stage8_oracle_benchmark.json").is_file()
