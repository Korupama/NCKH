from __future__ import annotations
import json
from pathlib import Path
import cv2
import numpy as np

from stage5_team_affiliation.gsr_benchmark import SoccerNetGSRDataset, evaluate_sequence, run_benchmark


def _dataset(tmp_path: Path):
    root = tmp_path / "GSR"; seq = root / "valid" / "SNGS-999"; seq.mkdir(parents=True)
    images=[]; anns=[]; aid=1
    H,W=180,320
    # 4 outfield per team + GK + referee. Team 0 yellow, team 1 white.
    specs=[]
    for tid in range(1,5): specs.append((tid,"player","left",30+tid*28,(0,200,230)))
    for tid in range(5,9): specs.append((tid,"player","right",160+(tid-5)*28,(0,0,230)))
    specs += [(9,"goalkeeper","left",20,(0,200,230)), (10,"referee",None,145,(30,30,30))]
    for fi in range(8):
        iid=str(1000+fi); fn=f"{fi:06d}.jpg"
        images.append({"image_id":iid,"file_name":fn,"width":W,"height":H,"is_labeled":True})
        im=np.full((H,W,3),(60,150,60),np.uint8)
        for tid,role,team,x,color in specs:
            x=int(x + (fi%2)); y=55; w=20; h=70
            # goalkeeper torso green-ish, lower body same as team to exercise GK affinity
            draw_color = (30,150,30) if role=="goalkeeper" else color
            cv2.rectangle(im,(x,y),(x+w,y+h),draw_color,-1)
            if role=="goalkeeper":
                cv2.rectangle(im,(x,y+35),(x+w,y+h),color,-1)
            anns.append({"id":str(aid),"image_id":iid,"track_id":tid,"supercategory":"object","category_id":1,
                         "attributes":{"role":role,"jersey":None,"team":team},
                         "bbox_image":{"x":x,"y":y,"w":w,"h":h,"x_center":x+w/2,"y_center":y+h/2}}); aid+=1
        cv2.imwrite(str(seq/fn),im)
    labels={"info":{"version":"1.3","frame_rate":5},"images":images,"annotations":anns,"categories":[]}
    (seq/"Labels-GameState.json").write_text(json.dumps(labels),encoding="utf-8")
    return root


def test_dataset_and_metrics(tmp_path):
    root=_dataset(tmp_path); ds=SoccerNetGSRDataset(root,"valid"); assert ds.discover()==["SNGS-999"]
    s=ds.load("SNGS-999"); assert len(s.team_gt_by_track())==9 and s.role_by_track()["10"]=="referee"
    # Perfect direct left/right labels.
    pred={tid:("left" if g==0 else "right") for tid,g in s.team_gt_by_track().items()}
    m=evaluate_sequence(s,pred,predictions_are_gt_labels=True)
    assert m["all_team_tracks"]["overall_accuracy"]==1.0
    assert m["referee_team_contamination_rate"]==0.0


def test_bbox_color_end_to_end(tmp_path):
    root=_dataset(tmp_path); out=tmp_path/"out"
    s=run_benchmark(dataset_root=root,split="valid",output_dir=out,method="bbox-color",progress_every=0)
    assert s["num_sequences"]==1
    assert s["groups"]["outfield"]["micro_overall_accuracy"] >= 0.75
    assert (out/"benchmark_summary.json").is_file()
    assert (out/"benchmark_summary.md").is_file()


def test_external_coco_like_predictions(tmp_path):
    root=_dataset(tmp_path); ds=SoccerNetGSRDataset(root,"valid"); seq=ds.load("SNGS-999")
    anns=[]
    for tid,g in seq.team_gt_by_track().items():
        anns.append({"track_id":int(tid),"attributes":{"team":"left" if g==0 else "right"}})
    pred_path=tmp_path/"pred.json"; pred_path.write_text(json.dumps({"annotations":anns}),encoding="utf-8")
    from stage5_team_affiliation.gsr_benchmark import load_external_predictions
    pred,direct=load_external_predictions(pred_path)
    m=evaluate_sequence(seq,pred,predictions_are_gt_labels=direct)
    assert direct is True and m["all_team_tracks"]["overall_accuracy"]==1.0
