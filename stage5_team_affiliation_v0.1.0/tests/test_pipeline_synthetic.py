import json, cv2, numpy as np
from pathlib import Path
from stage5_team_affiliation import Stage5Config, run_stage5

NAMES=['nose','left_eye','right_eye','left_ear','right_ear','left_shoulder','right_shoulder','left_elbow','right_elbow','left_wrist','right_wrist','left_hip','right_hip','left_knee','right_knee','left_ankle','right_ankle','left_big_toe','left_small_toe','left_heel','right_big_toe','right_small_toe','right_heel']+[f'k{i}' for i in range(23,133)]

def make_obs(fi,bbox,role):
    x1,y1,x2,y2=bbox; w=x2-x1; h=y2-y1
    coords={
      'left_shoulder':(x1+.28*w,y1+.25*h),'right_shoulder':(x1+.72*w,y1+.25*h),
      'left_hip':(x1+.34*w,y1+.55*h),'right_hip':(x1+.66*w,y1+.55*h),
      'left_knee':(x1+.38*w,y1+.78*h),'right_knee':(x1+.62*w,y1+.78*h),
    }
    kps=[]
    for i,n in enumerate(NAMES):
        p=coords.get(n)
        kps.append({'index':i,'name':n,'x':None if p is None else p[0],'y':None if p is None else p[1],'state':'MISSING' if p is None else 'VALID','raw_model_score':3.0 if p else None})
    return {'frame_index':fi,'source_bbox_xyxy':bbox,'keypoints_133':kps,'pose_status':'VALID'}

def build(tmp_path):
    W,H=640,360; fps=10; n=12
    vp=tmp_path/'v.mp4'; wr=cv2.VideoWriter(str(vp),cv2.VideoWriter_fourcc(*'mp4v'),fps,(W,H))
    tracks=[]
    specs=[]
    for i in range(4): specs.append((f'A{i}','player',[60+i*70,80,100+i*70,220],(20,20,220)))
    for i in range(4): specs.append((f'B{i}','player',[340+i*60,80,380+i*60,220],(220,30,20)))
    specs.append(('R0','referee',[300,230,335,350],(30,30,30)))
    # goalkeeper lower body matches A (red), torso green/different
    specs.append(('GK0','goalkeeper',[20,70,60,220],(40,200,40)))
    for fi in range(n):
        frame=np.zeros((H,W,3),np.uint8); frame[:]=(40,150,40)
        for tid,role,b,c in specs:
            x1,y1,x2,y2=b
            cv2.rectangle(frame,(x1,y1),(x2,y2),c,-1)
            if tid=='GK0':
                cv2.rectangle(frame,(x1,int(y1+.5*(y2-y1))),(x2,y2),(20,20,220),-1)
        wr.write(frame)
    wr.release()
    for tid,role,b,c in specs:
        tracks.append({'track_id':tid,'upstream_role':role,'upstream_role_score':.9,'upstream_identity_confidence':.95,'selected_frame_pose_status':'VALID','observations':[make_obs(fi,b,role) for fi in range(n)]})
    s3={'schema_version':'tracked-pose-2d-state-1.0','stage3_version':'test','coordinate_space':'RAW_DISTORTED_PIXEL','keypoint_schema':{'names':NAMES},'replay_context':{'video_path':str(vp),'image_width':W,'image_height':H,'selected_frame':6,'window_start':0,'window_end':11,'fps':fps},'tracks':tracks}
    sp=tmp_path/'s3.json'; sp.write_text(json.dumps(s3))
    return vp,sp

def test_pipeline(tmp_path):
    vp,sp=build(tmp_path)
    out=run_stage5(stage3_state=sp,video_path=vp,output_dir=tmp_path/'out',config=Stage5Config(sample_every_n_frames=1,min_cluster_margin=.01,goalkeeper_min_margin=.01))
    rec={r['track_id']:r for r in out['tracks']}
    assert rec['R0']['team_status']=='NOT_APPLICABLE'
    a={rec[f'A{i}']['team_id'] for i in range(4)}; b={rec[f'B{i}']['team_id'] for i in range(4)}
    assert len(a)==1 and len(b)==1 and a!=b
    assert out['metrics']['SelectedFrameTeamCoverage'] >= 8/9
    assert Path(out['artifacts']['selected_frame_overlay']).is_file()
