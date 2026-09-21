"""Re-select t0 inside existing replay caches; never overwrite source artifacts.

Reuses Stage1 cameras, Stage2 track/RTMW caches and Stage6 hybrid trajectory.
Rebuilds Stage3 selected-frame QA, Stage4 quality-gated ground anchors and Stage6
selected-frame contact output. It does not extend the cached temporal window.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path,data):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',default=r'D:\NCKH')
    parser.add_argument('--frame-index',type=int,required=True,help='Zero-based frame index')
    parser.add_argument('--output-dir',required=True,help='New directory, must not exist')
    args=parser.parse_args(); root=Path(args.root).resolve(); out=Path(args.output_dir).resolve(); t0=args.frame_index
    if out.exists(): raise ValueError('Output directory already exists; choose a new run directory')
    s1=root/'stage_1_camera_v12'; s3=root/'stage3_pose2d_v0.1'; s4=root/'stage4_metric3d_v0.2.0'; s6=root/'stage6_ball_localization_v0.4.4'
    for p in (s3,s4,s6): sys.path.insert(0,str(p))
    from stage3_pose2d import Stage3Config, run_stage3
    from stage4_metric3d.proxy_processor import run_stage4
    from stage4_metric3d.proxy_schemas import Stage4ProjectionConfig
    from ball_localization.camera import camera_state_for_frame, optimized_camera_dir
    from ball_localization.contracts import BallFrameState
    from ball_localization.pipeline import _candidate_from_dict, _selected_frame_payload
    from ball_localization.contact import refine_contact_file
    from ball_localization.visualization import render_selected_frame
    source3=s3/'runs/stage3_real_1920/tracked_pose_2d_state.json'
    source4=s4/'runs/stage4_real_1920_v031_quality_gated/metric_body_proxy_state.json'
    source6=s6/'outputs/stage6_v044_cached_replay/ball_trajectory_state.json'
    old3=read(source3); old4=read(source4); old6=read(source6)
    source2=Path(old3['source_stage2']['entity_state_path']); cache=Path(old3['source_stage2']['rtmw_cache_path'])
    entity=read(source2); rawcache=read(cache)
    sources=[source3,source4,source6,source2,cache]
    before={str(p):digest(p) for p in sources}
    for state in (entity,old3,old6):
        ctx=state['replay_context']
        if not int(ctx['window_start'])<=t0<=int(ctx['window_end']): raise ValueError('Target outside cached window: rerun upstream first')
    camera=camera_state_for_frame(s1,t0)
    if camera.status not in ('VALID','DEGRADED'): raise ValueError('Target camera is unusable')
    print(f'[Stage1] cached camera status={camera.status}; preserved without upgrading quality',flush=True)
    target=next((f for f in old6['frames'] if f['frame_index']==t0),None)
    if target is None: raise ValueError('Hybrid cache lacks target frame')
    out.mkdir(parents=True)
    # Reselect the existing stable player/GK population, requiring presence at new t0.
    # No identity, role, raw pose, detection or tracking result is changed.
    candidates=[]; selected_entities=[]
    for track in entity['tracks']:
        present=[o for o in track['observations'] if o['frame_index']==t0]
        eligible=bool(track.get('candidate_for_stage3')) and len(present)==1
        track['candidate_for_stage3']=eligible
        if eligible: candidates.append(track['track_id'])
        if present: selected_entities.append({'track_id':track['track_id'],'role':track['role'],'observation':present[0]})
    missing=[tid for tid in candidates if not any(o['frame_index']==t0 for o in rawcache.get('tracks',{}).get(tid,{}).get('observations',[]))]
    hand={'candidate_track_ids':candidates,'stage3_input_ready':not missing,
          'candidate_tracks_missing_rtmw_at_selected_frame':missing,'raw_rtmw_track_cache':str(cache),
          'entity_track_state_schema':entity['schema_version'],'note':'Reselected from original stable candidates present at new t0; no inference'}
    entity['replay_context']['selected_frame']=t0
    entity['selected_frame_entities']=selected_entities; entity['stage3_handoff']=hand
    entity['diagnostics']={'reselection':{'source_entity':str(source2),'old_selected_frame':old3['replay_context']['selected_frame'],
                           'selected_frame':t0,'original_metrics':entity.pop('metrics',{}),'original_status':entity.get('status')}}
    entity['status']='RESELECTED_CACHE'; entity['artifacts']={}
    ep=out/'stage2_reselected/stage2_entity_tracks.json'; hp=out/'stage2_reselected/stage3_handoff.json'
    write(ep,entity); write(hp,hand)
    print(f'[Reselect] frame={t0}; {len(candidates)} present candidates; missing RTMW={missing}',flush=True)
    cfg3=Stage3Config(**old3['configuration'])
    # Explicitly avoid new inference; reuse all cached image evidence and redo QA.
    cfg3.enable_fallback_reinference=False
    state3=run_stage3(entity_state=ep,rtmw_cache=cache,handoff=hp,output_dir=out/'stage3',config=cfg3)
    path3=out/'stage3/tracked_pose_2d_state.json'
    print('[Stage3] '+str(state3['metrics']['selected_frame_status_counts']),flush=True)
    state4=run_stage4(stage3_state=path3,camera_dir=optimized_camera_dir(s1),output_dir=out/'stage4_ground',
                      config=Stage4ProjectionConfig(**old4['configuration']))
    path4=out/'stage4_ground/stage4_downstream_handoff.json'
    print('[Stage4] ground anchors regenerated for new selected frame',flush=True)
    baseline=copy.deepcopy(old6); baseline['replay_context']['selected_frame']=t0
    frame=copy.deepcopy(target); frame['candidate']=_candidate_from_dict(frame.get('candidate'))
    baseline['selected_frame_ball']=_selected_frame_payload([BallFrameState(**frame)],t0,baseline['ball_radius_m'])
    baseline['status']='VALID' if baseline['selected_frame_ball']['usable_for_offside_longitudinal_coordinate'] else 'DEGRADED'
    baseline['diagnostics']['selected_frame_reselection']={'source':str(source6),'new_t0':t0,'cached_window_preserved':True,'geometry_recomputed':False}
    basepath=out/'stage6_hybrid/ball_trajectory_state.json'; baseline['artifacts']={'ball_trajectory_state':str(basepath)}
    write(basepath,baseline)
    render_selected_frame(baseline['replay_context']['video_path'],baseline['selected_frame_ball'],out/'stage6_hybrid/selected_frame_ball.png')
    result=refine_contact_file(basepath,s1,path3,path4,out/'stage6_contact')
    assert result['replay_context']['selected_frame']==t0
    assert read(path4)['selected_frame']==t0 and state3['replay_context']['selected_frame']==t0
    assert [f for f in baseline['frames'] if f['frame_index']!=t0]==[f for f in result['frames'] if f['frame_index']!=t0]
    assert result['candidates_by_frame']==old6['candidates_by_frame']
    assert all(digest(p)==h for p,h in before.items()),'Source input mutated'
    selected=result['selected_frame_ball']
    summary={'frame_index':t0,'frame_index_base':0,'time_seconds':t0/entity['replay_context']['fps'],
             'camera_status':camera.status,'cached_window':[entity['replay_context']['window_start'],entity['replay_context']['window_end']],
             'source_hashes':before,'source_artifacts_unchanged':True,'candidate_tracks':candidates,
             'stage3_metrics':state3['metrics'],'stage4_readiness':read(path4)['readiness'],
             'selected_frame_ball':selected,'research_accuracy_frozen':False,
             'limitations':['Existing cache window preserved, not recentered','Stable candidate roles retained from Stage2',
                            'Only Stage4 distal-foot ground handoff used; no full human 3D rerun']}
    write(out/'reselection_report.json',summary)
    print(json.dumps({'output_dir':str(out),'frame_index':t0,'contact':selected['contact'],
                     'method':selected['method'],'xyz':selected['center_xyz_world_m'],'fallback':selected['fallback']},indent=2))

if __name__=='__main__': main()
