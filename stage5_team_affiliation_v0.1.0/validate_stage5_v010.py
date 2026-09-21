from __future__ import annotations

import json, tempfile
from pathlib import Path
import subprocess, sys


def main():
    root=Path(__file__).resolve().parent
    cmd=[sys.executable,'-m','pytest','-q',str(root/'tests')]
    p=subprocess.run(cmd,cwd=root,text=True,capture_output=True)
    report={
      'schema_version':'stage5-v010-validation-1.0',
      'stage5_version':'stage5-team-affiliation-0.1.0',
      'pytest_returncode':p.returncode,
      'pytest_stdout':p.stdout.strip(),
      'pytest_stderr':p.stderr.strip(),
      'status':'PASS' if p.returncode==0 else 'FAIL',
      'accuracy_claim':False,
      'note':'Implementation/synthetic validation only. Real SoccerNet-GSR team accuracy is NOT_EVALUATED until benchmark images are run.'
    }
    out=root/'validation_reports'/'STAGE5_V010_VALIDATION_REPORT.json'
    out.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
    raise SystemExit(p.returncode)

if __name__=='__main__': main()
