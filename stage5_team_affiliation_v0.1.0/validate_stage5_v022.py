"""Synthetic/unit validator only. No replay, GSR images, or calibration run."""
import json
from pathlib import Path
import subprocess
import sys


def main():
    root = Path(__file__).resolve().parent
    process = subprocess.run(
        [sys.executable, '-m', 'pytest', '-q', str(root / 'tests/test_residual_v020.py')],
        cwd=root, capture_output=True, text=True,
    )
    report = {
        'schema_version': 'stage5-residual-validation-1.2',
        'package_version': '0.2.2',
        'status': 'PASS' if process.returncode == 0 else 'FAIL',
        'stdout': process.stdout, 'stderr': process.stderr,
        'scope': 'residual appearance recovery, TRAIN leakage guard, calibration search, regression',
        'real_pipeline_executed': False, 'calibration_executed': False,
        'research_accuracy_frozen': False,
    }
    directory = root / 'validation_reports'; directory.mkdir(exist_ok=True)
    (directory / 'STAGE5_V022_VALIDATION_REPORT.json').write_text(
        json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
    raise SystemExit(process.returncode)


if __name__ == '__main__':
    main()
