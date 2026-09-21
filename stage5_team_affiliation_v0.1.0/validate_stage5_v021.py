"""Small v0.2.1 synthetic/unit validator. No replay or GSR image pipeline."""
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
        'schema_version': 'stage5-residual-validation-1.1',
        'package_version': '0.2.1',
        'status': 'PASS' if process.returncode == 0 else 'FAIL',
        'stdout': process.stdout,
        'stderr': process.stderr,
        'scope': 'v0.2.1 candidate-only pairwise gate, fail-closed roles, mocked orchestration',
        'real_pipeline_executed': False,
        'research_accuracy_frozen': False,
    }
    directory = root / 'validation_reports'
    directory.mkdir(exist_ok=True)
    (directory / 'STAGE5_V021_VALIDATION_REPORT.json').write_text(
        json.dumps(report, indent=2), encoding='utf-8'
    )
    print(json.dumps(report, indent=2))
    raise SystemExit(process.returncode)


if __name__ == '__main__':
    main()
