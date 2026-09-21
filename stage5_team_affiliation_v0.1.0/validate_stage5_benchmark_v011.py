from __future__ import annotations
import json, tempfile
from pathlib import Path
import cv2
import numpy as np

from stage5_team_affiliation.gsr_benchmark import SoccerNetGSRDataset, run_benchmark


def make_synthetic(root: Path) -> Path:
    seq=root/'valid'/'SNGS-999'; seq.mkdir(parents=True)
    images=[]; anns=[]; aid=1; H,W=180,320
    specs=[]
    for tid in range(1,5): specs.append((tid,'player','left',30+tid*28,(0,200,230)))
    for tid in range(5,9): specs.append((tid,'player','right',160+(tid-5)*28,(0,0,230)))
    specs += [(9,'goalkeeper','left',20,(0,200,230)), (10,'referee',None,145,(30,30,30))]
    for fi in range(8):
        iid=str(1000+fi); fn=f'{fi:06d}.jpg'; images.append({'image_id':iid,'file_name':fn,'width':W,'height':H,'is_labeled':True})
        im=np.full((H,W,3),(60,150,60),np.uint8)
        for tid,role,team,x,color in specs:
            x=int(x+(fi%2)); y=55; w=20; h=70
            draw=(30,150,30) if role=='goalkeeper' else color
            cv2.rectangle(im,(x,y),(x+w,y+h),draw,-1)
            if role=='goalkeeper': cv2.rectangle(im,(x,y+35),(x+w,y+h),color,-1)
            anns.append({'id':str(aid),'image_id':iid,'track_id':tid,'supercategory':'object','category_id':1,
                         'attributes':{'role':role,'jersey':None,'team':team},
                         'bbox_image':{'x':x,'y':y,'w':w,'h':h,'x_center':x+w/2,'y_center':y+h/2}}); aid+=1
        cv2.imwrite(str(seq/fn),im)
    (seq/'Labels-GameState.json').write_text(json.dumps({'info':{'version':'1.3','frame_rate':5},'images':images,'annotations':anns,'categories':[]}),encoding='utf-8')
    return root


def main():
    tmp=Path(tempfile.mkdtemp(prefix='stage5_bench_'))
    root=make_synthetic(tmp/'GSR')
    ds=SoccerNetGSRDataset(root,'valid')
    inspect=ds.inspect(1)
    out=tmp/'out'
    summary=run_benchmark(dataset_root=root,split='valid',output_dir=out,method='bbox-color',progress_every=0)
    allm=summary['groups']['all_team_tracks']; outm=summary['groups']['outfield']
    checks={
        'dataset_discovery': inspect['num_sequences']==1,
        'synthetic_outfield_selective_accuracy_1': abs(float(outm['micro_selective_accuracy'])-1.0)<1e-9,
        'synthetic_outfield_overall_ge_075': float(outm['micro_overall_accuracy'])>=0.75,
        'referee_contamination_zero': float(summary['referee']['team_contamination_rate'])==0.0,
        'summary_json_written': (out/'benchmark_summary.json').is_file(),
        'summary_md_written': (out/'benchmark_summary.md').is_file(),
    }
    report={'schema_version':'stage5-benchmark-validator-1.0','package_version':'0.1.1','checks':checks,'summary':summary,'status':'PASS' if all(checks.values()) else 'FAIL'}
    target=Path('validation_reports/STAGE5_GSR_BENCHMARK_V011_VALIDATION.json')
    target.parent.mkdir(parents=True,exist_ok=True); target.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
    if report['status']!='PASS': raise SystemExit(1)

if __name__=='__main__': main()
