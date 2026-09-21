from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from stage2_entities.benchmark.soccernet_gsr import SoccerNetGSRDataset
from stage2_entities.benchmark.protocols import BenchmarkProtocol, build_protocol_windows
from stage2_entities.benchmark.metrics_detection import evaluate_selected_frame
from stage2_entities.benchmark.metrics_tracking import evaluate_tracking_window
from stage2_entities.benchmark.runner import BenchmarkConfig, run_benchmark


def _make_dataset(tmp_path: Path):
    root=tmp_path/"SoccerNetGS"; seq=root/"valid"/"SNGS-001"; seq.mkdir(parents=True)
    images=[]; anns=[]; aid=1
    for fi in range(21):
        iid=str(100000+fi)
        images.append({"image_id":iid,"file_name":f"{fi:06d}.jpg","width":320,"height":180,"is_labeled":True})
        objs=[
            (1,"player",[20+fi,50,20,60]),
            (2,"goalkeeper",[240-fi,55,22,58]),
            (3,"referee",[130,52+0.2*fi,20,60]),
        ]
        for tid,role,(x,y,w,h) in objs:
            anns.append({"id":str(aid),"image_id":iid,"track_id":tid,"supercategory":"object","category_id":1,
                         "attributes":{"role":role,"jersey":"","team":""},
                         "bbox_image":{"x":x,"y":y,"w":w,"h":h,"x_center":x+w/2,"y_center":y+h/2}})
            aid+=1
    labels={"info":{"version":"1.3","seq_length":21,"frame_rate":5},"images":images,"annotations":anns,"categories":[]}
    (seq/"Labels-GameState.json").write_text(json.dumps(labels),encoding="utf-8")
    return root


def _make_perception_cache(output: Path):
    pdir=output/"sequences"/"SNGS-001"/"perception"; fdir=pdir/"frame_perception"; fdir.mkdir(parents=True)
    records=[]
    for fi in range(21):
        specs=[
            ("human_001",2,"Player","player",[20+fi,50,40+fi,110]),
            ("human_002",3,"Goalkeeper","goalkeeper",[240-fi,55,262-fi,113]),
            ("human_003",4,"Main referee","referee",[130,52+0.2*fi,150,112+0.2*fi]),
        ]
        raw=[]; humans=[]
        for j,(hid,lid,label,role,box) in enumerate(specs,1):
            did=f"{role}_{j:03d}"
            raw.append({"detection_id":did,"label_id":lid,"label":label,"score":0.99,"bbox_xyxy":box})
            ev={r:0.0 for r in ("player","goalkeeper","referee","other")}; ev[role]=0.99
            humans.append({"physical_human_id":hid,"representative_detection_id":did,"representative_label_id":lid,
                           "representative_label":label,"bbox_xyxy":box,"detector_score":0.99,"source_detection_ids":[did],
                           "class_evidence":{label:0.99},"role_evidence":ev,"resolved_role":role,"role_score":0.99,
                           "role_margin":0.99,"role_status":"VALID","pose_required":role in {"player","goalkeeper"},
                           "candidate_hint":role in {"player","goalkeeper"},"consolidation_reason":"single_detection"})
        payload={"frame_index":fi,"image_path":"synthetic","image_width":320,"image_height":180,"raw_detections":raw,
                 "raw_low_score_detections":raw,"humans":humans,"rescue_humans":[],
                 "pose_cache":{},"ball_detections":[],"timing_seconds":{},"provenance":{"synthetic":True}}
        path=fdir/f"frame_{fi:09d}_perception.json"; path.write_text(json.dumps(payload),encoding="utf-8")
        records.append({"frame_index":fi,"perception_json":str(path.resolve())})
    manifest={"schema_version":"stage2-sst-rtmw-perception-manifest-1.0","stage2_version":"stage2-sst-rtmw-1.3.0","models":{"synthetic":True},"configuration":{
        "class_thresholds": {
            "Ball": 0.35, "Player": 0.50, "Goalkeeper": 0.50,
            "Main referee": 0.55, "Side referee": 0.55, "Staff members": 0.60
        },
        "person_nms_iou": 0.65,
        "ball_nms_iou": 0.30,
        "consolidation_high_iou": 0.82,
        "consolidation_low_iou": 0.60,
        "consolidation_max_center_distance": 0.18,
        "consolidation_min_area_similarity": 0.65,
        "consolidation_min_intersection_over_min": 0.75,
        "raw_human_score_floor": 0.20,
        "rescue_overlap_with_active_iou": 0.65,
        "role_ambiguous_margin": 0.05,
        "keypoint_threshold": 1.0
    },"frames":records}
    (pdir/"perception_manifest.json").write_text(json.dumps(manifest),encoding="utf-8")


def test_loader_protocol(tmp_path):
    root=_make_dataset(tmp_path)
    ds=SoccerNetGSRDataset(root,"valid")
    assert ds.discover()==["SNGS-001"]
    seq=ds.load("SNGS-001")
    assert seq.version=="1.3" and seq.fps==5 and seq.num_frames==21
    assert {x.role for x in seq.gt(10)}=={"player","goalkeeper","referee"}
    windows=build_protocol_windows(seq,BenchmarkProtocol.named("quick",half_window_seconds=1.0))
    assert [w.target_frame for w in windows]==[5,10,15]
    assert windows[0].window_start==0 and windows[-1].window_end==20


def test_tracking_metrics_perfect_and_swap():
    frames=list(range(5))
    gt=[]; pred=[]
    for fi in frames:
        gt.extend([
            {"frame_index":fi,"track_id":"g1","bbox_xyxy":[10+fi,10,30+fi,50]},
            {"frame_index":fi,"track_id":"g2","bbox_xyxy":[100-fi,10,120-fi,50]},
        ])
        # Perfect until frame 2, then swap prediction identity labels while boxes stay correct.
        pred.extend([
            {"frame_index":fi,"track_id":"p1" if fi<3 else "p2","bbox_xyxy":[10+fi,10,30+fi,50]},
            {"frame_index":fi,"track_id":"p2" if fi<3 else "p1","bbox_xyxy":[100-fi,10,120-fi,50]},
        ])
    perfect=[]
    for fi in frames:
        perfect.extend([
            {"frame_index":fi,"track_id":"p1","bbox_xyxy":[10+fi,10,30+fi,50]},
            {"frame_index":fi,"track_id":"p2","bbox_xyxy":[100-fi,10,120-fi,50]},
        ])
    a=evaluate_tracking_window(gt,perfect,frames,target_frame=2)
    assert abs(a["HOTA"]-1)<1e-9 and abs(a["IDF1"]-1)<1e-9 and abs(a["TCR"]-1)<1e-9
    b=evaluate_tracking_window(gt,pred,frames,target_frame=2)
    assert b["AssA"]<1 and b["IDF1"]<1 and b["TCR"]<1


def test_full_runner_from_cache(tmp_path):
    root=_make_dataset(tmp_path); out=tmp_path/"bench"; _make_perception_cache(out)
    cfg=BenchmarkConfig(dataset_root=root,split="valid",output_dir=out,protocol=BenchmarkProtocol.named("quick",half_window_seconds=1.0),
                        experiments=("primary","no_pose","oracle_boxes_geom"))
    summary=run_benchmark(cfg)
    assert summary["detection"]["CandidateRecall"]==1.0
    assert summary["detection"]["RefereeLeakageRate"]==0.0
    assert summary["tracking_candidate"]["HOTA"]>0.999999
    assert summary["tracking_candidate"]["IDF1"]>0.999999
    assert summary["tracking_candidate"]["TCR"]>0.999999
    assert (out/"benchmark_summary.json").is_file()
    assert (out/"window_metrics.csv").is_file()
    assert (out/"failure_cases.json").is_file()


def test_cache_version_guard(tmp_path):
    from stage2_entities.benchmark.runner import _cache_status
    cache = tmp_path / "perception_manifest.json"
    cache.write_text(json.dumps({
        "schema_version": "stage2-sst-rtmw-perception-manifest-1.0",
        "stage2_version": "stage2-sst-rtmw-0.9.0",
        "models": {},
        "frames": [{"frame_index": 1, "perception_json": "x"}],
    }), encoding="utf-8")
    status, reason, _ = _cache_status(cache, [1], {})
    assert status == "incompatible"
    assert "stage2_version" in reason


def test_benchmark_checkpoint_interrupt_and_resume(tmp_path):
    root = _make_dataset(tmp_path)
    out = tmp_path / "bench_resume"
    _make_perception_cache(out)
    cfg_interrupt = BenchmarkConfig(
        dataset_root=root,
        split="valid",
        output_dir=out,
        protocol=BenchmarkProtocol.named("quick", half_window_seconds=1.0),
        experiments=("primary", "no_pose"),
        quiet=True,
        debug_interrupt_after_new_eval_units=2,
    )
    import pytest
    with pytest.raises(KeyboardInterrupt):
        run_benchmark(cfg_interrupt)

    state_path = out / "checkpoints" / "benchmark_state.json"
    assert state_path.is_file()
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["status"] == "INTERRUPTED"
    assert state["completed_eval_units"] == 2

    # Mark the first detection checkpoint. A correct resume must reuse it rather than rewrite it.
    det_paths = sorted((out / "checkpoints" / "evaluation").glob("*/t*/detection.json"))
    assert det_paths
    det = json.loads(det_paths[0].read_text(encoding="utf-8"))
    det["resume_sentinel"] = "keep-me"
    det_paths[0].write_text(json.dumps(det), encoding="utf-8")

    cfg_resume = BenchmarkConfig(
        dataset_root=root,
        split="valid",
        output_dir=out,
        protocol=BenchmarkProtocol.named("quick", half_window_seconds=1.0),
        experiments=("primary", "no_pose"),
        quiet=True,
    )
    summary = run_benchmark(cfg_resume)
    final_state = json.loads(state_path.read_text(encoding="utf-8"))
    assert final_state["status"] == "COMPLETE"
    assert final_state["completed_eval_units"] == 9  # 3 windows * (detection + 2 experiments)
    assert json.loads(det_paths[0].read_text(encoding="utf-8"))["resume_sentinel"] == "keep-me"
    assert summary["tracking_candidate"]["HOTA"] > 0.999999
    assert (out / "checkpoints" / "partial_summary.json").is_file()


def test_threshold_sweep_reuses_low_score_cache(tmp_path):
    from stage2_entities.benchmark.threshold_sweep import run_threshold_sweep
    root=_make_dataset(tmp_path); out=tmp_path/"bench_sweep"; _make_perception_cache(out)
    cfg=BenchmarkConfig(
        dataset_root=root,split="valid",output_dir=out,
        protocol=BenchmarkProtocol.named("quick",half_window_seconds=1.0),
        experiments=("primary",),quiet=True,
    )
    run_benchmark(cfg)
    report=run_threshold_sweep(
        dataset_root=root,split="valid",benchmark_dir=out,
        player_thresholds=(0.5,),goalkeeper_thresholds=(0.5,),quiet=True,
    )
    assert report["recommended"]["CandidateRecall"] == 1.0
    assert report["recommended"]["CandidatePrecision"] == 1.0
    assert (out/"threshold_sweep"/"threshold_sweep.csv").is_file()
