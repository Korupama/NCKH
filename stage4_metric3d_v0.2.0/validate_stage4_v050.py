from __future__ import annotations

import json
import tempfile
from pathlib import Path
import numpy as np

from tests_v05.helpers import build_case
from stage4_metric3d.backends.sam3d_pitch_refined.pipeline import preflight_v05, run_v05


def main() -> int:
    with tempfile.TemporaryDirectory(prefix='stage4_v050_') as td:
        root=Path(td); s3,cams,cache,true_trans,frames=build_case(root/'case')
        pre=preflight_v05(stage3_state=s3,camera_dir=cams,sam3d_cache=cache)
        direct=run_v05(stage3_state=s3,camera_dir=cams,sam3d_cache=cache,output_dir=root/'direct',refine=False)
        refined=run_v05(stage3_state=s3,camera_dir=cams,sam3d_cache=cache,output_dir=root/'refined',refine=True)
        od=next(o for o in direct['tracks'][0]['observations'] if o['frame_index']==11)
        orf=next(o for o in refined['tracks'][0]['observations'] if o['frame_index']==11)
        e0=float(np.linalg.norm(np.asarray(od['translation']['refined_cam_m'])-true_trans[1]))
        e1=float(np.linalg.norm(np.asarray(orf['translation']['refined_cam_m'])-true_trans[1]))
        report={
            'schema_version':'stage4-v050-validation-1.0','preflight_ready':pre['ready'],
            'camera_convention':pre['camera_convention'], 'direct_translation_error_m':e0,
            'refined_translation_error_m':e1,'improved':bool(e1<e0),
            'output_schema':refined['schema_version'],'handoff_exists':(root/'refined'/'stage4_downstream_handoff.json').is_file(),
            'quality_gates':refined['quality_gates'],
        }
        report['status']='PASS' if report['preflight_ready'] and report['improved'] and report['output_schema']=='world-grounded-pose-state-1.1' else 'FAIL'
    out=Path('validation_reports/STAGE4_V050_SYNTHETIC_VALIDATION_REPORT.json'); out.parent.mkdir(exist_ok=True); out.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2)); return 0 if report['status']=='PASS' else 2


if __name__=='__main__': raise SystemExit(main())
