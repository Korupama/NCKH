"""Release hotfix regression; no real-world accuracy claim."""
from pathlib import Path
import subprocess
import sys
if __name__=='__main__':
    raise SystemExit(subprocess.call([sys.executable,'-m','pytest',str(Path(__file__).parent/'tests/test_contact_v051.py'),'-q','-p','no:cacheprovider']))
