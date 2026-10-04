"""Conservative image-space contact hypotheses, not a trained contact classifier.

Only distal-foot ground anchors are consumed. No Stage-4 upper-body XYZ is read.
Temporal proximity is evidence of possible contact, not proof of physical touch.
"""
from __future__ import annotations
import copy
import json
from pathlib import Path
import cv2
import numpy as np


def validate_replay(reference, other):
    for key in ('image_width', 'image_height', 'selected_frame', 'fps', 'coordinate_space'):
        if key not in other or other[key] != reference.get(key):
            raise ValueError(f'Replay mismatch: {key}')
    a, b = reference.get('video_path'), other.get('video_path')
    if not a or not b or str(a).replace('\\', '/').lower() != str(b).replace('\\', '/').lower():
        raise ValueError('Replay mismatch: video_path')


def segment_distance(p, a, b):
    delta = b-a
    t = np.clip(np.dot(p-a, delta)/max(float(np.dot(delta, delta)), 1e-12), 0, 1)
    return float(np.linalg.norm(p-a-t*delta))


def region_distances(observation, uv):
    points = {k['name']: np.array([k['x'], k['y']], float)
              for k in observation.get('keypoints_133', [])
              if k.get('state') == 'VALID' and k.get('x') is not None and k.get('y') is not None
              and np.isfinite([k['x'], k['y']]).all()}
    p = np.asarray(uv, float)
    out = {}
    def point_group(region, names):
        ds = [float(np.linalg.norm(p-points[n])) for n in names if n in points]
        if ds: out[region] = min(ds)
    point_group('HEAD', ['nose','left_eye','right_eye','left_ear','right_ear'])
    point_group('KNEE', ['left_knee','right_knee'])
    for region, pairs in {
        'THIGH': [('hip','knee')], 'LOWER_LEG': [('knee','ankle')],
        'FOOT': [('ankle','heel'),('heel','big_toe'),('big_toe','small_toe')],
        'ARM_HAND': [('elbow','wrist')],
    }.items():
        ds=[]
        for side in ('left','right'):
            for a,b in pairs:
                a,b=f'{side}_{a}',f'{side}_{b}'
                if a in points and b in points: ds.append(segment_distance(p,points[a],points[b]))
        if ds: out[region]=min(ds)
    names=['left_shoulder','right_shoulder','right_hip','left_hip']
    if all(n in points for n in names):
        polygon=np.array([points[n] for n in names], np.float32)
        out['TORSO']=max(0., -float(cv2.pointPolygonTest(polygon, tuple(map(float,p)), True)))
    return out


def associate_contact(poses, frames, t0):
    selected={int(f['frame_index']): f.get('candidate') for f in frames}
    candidate=selected.get(t0)
    if not candidate: return {'status':'NO_BALL_OBSERVATION','track_id':None,'region':'UNKNOWN'}
    radius=max(2.,float(candidate.get('diameter_px', 4))/2)
    threshold=radius+4.  # fixed pixel tolerance, uncalibrated
    hypotheses=[]
    for track in poses.get('tracks', []):
        if track.get('upstream_role','player').lower() not in ('player','goalkeeper'): continue
        obs={int(o['frame_index']):o for o in track.get('observations', [])}
        if t0 not in obs: continue
        distances=region_distances(obs[t0], candidate['center_uv'])
        if not distances: continue
        region=min(distances,key=distances.get)
        distance=distances[region]
        history={}
        for fi in range(t0-3,t0+4):
            if fi in obs and selected.get(fi):
                ds=region_distances(obs[fi],selected[fi]['center_uv'])
                if region in ds: history[fi]=ds[region]
        before=[v for fi,v in history.items() if fi<t0]
        after=[v for fi,v in history.items() if fi>t0]
        temporal=bool(before and after and max(before)>distance+1 and max(after)>distance+1
                      and distance<=min(history.values())+2)
        hypotheses.append({'track_id':track['track_id'],'region':region,'image_distance_px':distance,
                           'threshold_px':threshold,'temporal_contact_evidence':temporal,
                           'distance_by_frame':history,'region_distances_px':distances})
    hypotheses.sort(key=lambda x:x['image_distance_px'])
    if not hypotheses: return {'status':'NO_POSE_SUPPORT','track_id':None,'region':'UNKNOWN'}
    best=copy.deepcopy(hypotheses[0])
    eligible=[h for h in hypotheses if h['image_distance_px']<=threshold]
    best['status']=('NO_CONTACT_EVIDENCE' if not eligible else
                    'AMBIGUOUS' if len(eligible)>1 and eligible[1]['image_distance_px']-best['image_distance_px']<3 else
                    'SUPPORTED' if best['temporal_contact_evidence'] else 'INSUFFICIENT_TEMPORAL_SUPPORT')
    best['hypotheses']=hypotheses[:5]
    best['evidence_semantics']='image-space heuristic; not calibrated contact probability'
    if best['status']!='SUPPORTED':
        best['nearest_track_id']=best['track_id']; best['track_id']=None
    return best


def ground_anchor(handoff, track_id, t0):
    if handoff is None:
        return {'status':'MISSING','xy_world_m':None,'source':None}
    if handoff.get('selected_frame')!=t0: raise ValueError('Stage4 selected_frame mismatch')
    # SAM3D's pose-only handoff has no quality-gated distal-foot anchor.
    # Missing optional evidence is not a coordinate-convention violation.
    track = next((t for t in handoff.get('tracks',[]) if track_id is not None and t.get('track_id')==track_id), None)
    if track is None or not track.get('selected_frame_ground_anchor'):
        return {'status':'MISSING','xy_world_m':None,'source':None,
                'upstream_reason':'NO_QUALITY_GATED_GROUND_ANCHOR'}
    axes=handoff.get('coordinate_frame',{})
    # This adapter deliberately accepts the quality-gated distal-foot contract only.
    if axes.get('units') not in ('m','meters','metres'): raise ValueError('Stage4 metric units required')
    for key,value in {'origin':'center','x_axis':'goal_to_goal','y_axis':'touchline_to_touchline','z_axis':'up'}.items():
        if axes.get(key)!=value: raise ValueError(f'Stage4 convention mismatch: {key}')
    for track in handoff.get('tracks',[]):
        if track.get('track_id')!=track_id: continue
        a=track.get('selected_frame_ground_anchor') or {}
        xyz=a.get('xyz_ground_m')
        method=a.get('method','')
        allowed=method=='DENSEST_ANKLE_FOOT_CLUSTER_MEDIAN'
        valid=a.get('status')=='VALID' and allowed and xyz is not None and len(xyz)==3 and np.isfinite(xyz).all() and abs(xyz[2])<1e-6
        return {'status':'VALID' if valid else 'UNUSABLE','xy_world_m':xyz[:2] if valid else None,
                'source':method,'upstream_status':a.get('status'),'upstream_reason':a.get('reason')}
    return {'status':'MISSING','xy_world_m':None,'source':None}


def plane_sensitivity(camera, uv, y):
    nominal=camera.intersect_y_plane(uv,y)[0]
    samples=[]
    for delta in (-.5,-.25,.25,.5):
        xyz=camera.intersect_y_plane(uv,y+delta)[0]
        samples.append({'delta_y_m':delta,'delta_x_m':float(xyz[0]-nominal[0]) if np.isfinite(xyz).all() else None})
    values=[abs(s['delta_x_m']) for s in samples if s['delta_x_m'] is not None]
    return {'samples':samples,'max_abs_delta_x_m':max(values) if values else None,
            'median_abs_delta_x_m':float(np.median(values)) if values else None,
            'P90_abs_delta_x_m':float(np.percentile(values,90)) if values else None,
            'P95_abs_delta_x_m':float(np.percentile(values,95)) if values else None,
            'semantics':'deterministic sensitivity, not a confidence interval'}


def apply_contact(source, poses, handoff, camera):
    validate_replay(source['replay_context'],poses['replay_context'])
    result=copy.deepcopy(source); t0=int(source['replay_context']['selected_frame'])
    if (camera.image_width,camera.image_height)!=(source['replay_context']['image_width'],source['replay_context']['image_height']):
        raise ValueError('Camera image dimensions mismatch')
    contact=associate_contact(poses,source['frames'],t0)
    is_foot=contact.get('region')=='FOOT'
    supported=contact['status'] in ('SUPPORTED','VALID')
    # An unconfirmed toucher cannot supply a contact-plane anchor.
    anchor=({'status':'NOT_REQUIRED','xy_world_m':None,'source':None} if is_foot or not supported
            else ground_anchor(handoff,contact.get('track_id'),t0))
    frame=next(f for f in result['frames'] if int(f['frame_index'])==t0)
    old=source['selected_frame_ball']; candidate=frame.get('candidate'); radius=float(source.get('ball_radius_m',.11))
    xyz=None; method=None; rejection=None; sensitivity=None
    camera_usable=camera.status in ('VALID','DEGRADED') if is_foot else camera.status=='VALID'
    if not supported:
        rejection=contact['status']
    elif not camera_usable:
        rejection='GROUND_CAMERA_UNUSABLE' if is_foot else 'VERTICAL_CAMERA_REQUIRES_VALID'
    elif not is_foot and anchor['status']!='VALID':
        rejection='VERTICAL_GROUND_ANCHOR_UNUSABLE'
    if rejection is None:
        uv=candidate['center_uv']; region=contact['region']
        proposed=(camera.intersect_z_plane(uv,radius)[0] if region=='FOOT' else camera.intersect_y_plane(uv,anchor['xy_world_m'][1])[0])
        height_limits={'FOOT':(.05,.3),'LOWER_LEG':(.05,1.),'KNEE':(.1,1.3),'THIGH':(.2,1.8),'TORSO':(.3,2.4),'HEAD':(.5,2.8),'ARM_HAND':(.05,2.8)}
        low,high=height_limits.get(region,(0,0))
        half_length=float(camera.pitch.get('length_m',105.0))/2
        half_width=float(camera.pitch.get('width_m',68.0))/2
        plausible=np.isfinite(proposed).all() and low<=proposed[2]<=high and abs(proposed[0])<=half_length and abs(proposed[1])<=half_width
        # Only the vertical plane depends on the player's ground anchor.
        if not is_foot:
            plausible=plausible and np.linalg.norm(proposed[:2]-np.asarray(anchor['xy_world_m']))<=1.5
        if plausible:
            xyz=proposed.tolist(); method='CONTACT_GROUND_PLANE' if region=='FOOT' else 'CONTACT_VERTICAL_PLANE'
            sensitivity=plane_sensitivity(camera,uv,anchor['xy_world_m'][1]) if region!='FOOT' else None
        else: rejection='GROUND_INTERSECTION_INVALID_OR_OUTSIDE_PITCH' if is_foot else 'VERTICAL_CONTACT_GEOMETRY_IMPLAUSIBLE'
    fallback=xyz is None
    if fallback:
        xyz=old.get('center_xyz_world_m'); method=old.get('method')
        if xyz is None and frame.get('size_prior_xyz_world_m') is not None:
            size=np.asarray(frame['size_prior_xyz_world_m'],float)
            if np.isfinite(size).all() and .055<=size[2]<=15 and abs(size[0])<=52.5 and abs(size[1])<=34:
                xyz=size.tolist(); method='MONOCULAR_BALL_SIZE_PRIOR_FALLBACK'
    usable=xyz is not None and np.isfinite(xyz).all()
    if not usable: xyz=None; method=None
    status=(('DEGRADED_' if camera.status=='DEGRADED' else 'VALID_')+method if not fallback
            else 'DEGRADED_CONTACT_FALLBACK' if usable else 'MISSING')
    selected=copy.deepcopy(old)
    for key in ('X_world_m','Y_world_m','Z_world_m','ball_center_x_extent_m'): selected.pop(key,None)
    selected.update(status=status,center_xyz_world_m=xyz,method=method,camera_status=camera.status,usable_for_offside_longitudinal_coordinate=bool(usable))
    extent=[xyz[0]-radius,xyz[0]+radius] if usable else None
    if usable: selected.update(X_world_m=xyz[0],Y_world_m=xyz[1],Z_world_m=xyz[2],ball_center_x_extent_m=extent)
    if not fallback and is_foot:
        # Do not retain stale cached ground fields beside newly projected XYZ.
        for payload in (selected,frame):
            payload['ground_center_xyz_world_m']=list(xyz)
            payload['ground_contact_xyz_world_m']=[xyz[0],xyz[1],0.0]
    selected.update(observation=copy.deepcopy(candidate),contact=contact,ground_anchor=anchor,
        localization={'selected_method':method,'center_xyz_world_m':xyz,'X_world_m':xyz[0] if usable else None,
                      'ball_center_x_extent_m':extent,'usable_for_offside':bool(usable),'accuracy_validated':False,
                      'ground_anchor_required':not is_foot,'ground_anchor_used':not is_foot and not fallback,
                      'camera_status':camera.status},
        fallback={'hybrid_v044_available':old.get('center_xyz_world_m') is not None,'used':fallback,'reason':rejection,
                  'baseline_method':old.get('method'),'baseline_xyz_world_m':old.get('center_xyz_world_m')},
        anchor_sensitivity=sensitivity)
    if selected['observation']:
        selected['observation']['sources']=candidate.get('metadata',{}).get('sources',[candidate.get('source')])
    selected['delta_x_vs_hybrid_m']=(xyz[0]-old['center_xyz_world_m'][0]) if usable and old.get('center_xyz_world_m') is not None else None
    frame.update(selected_center_xyz_world_m=xyz,selected_method=method,localization_status=status)
    frame.setdefault('diagnostics',{})['contact_aware']=copy.deepcopy(selected['contact'])
    from .version import STAGE6_VERSION
    result.update(schema_version='ball-trajectory-state-1.2',stage6_version=STAGE6_VERSION,
                  localization_mode='contact-aware',selected_frame_ball=selected,status='DEGRADED' if fallback or camera.status!='VALID' else 'VALID')
    result.setdefault('diagnostics',{}).pop('v050_gates',None)
    result['diagnostics']['v051_gates']={'implementation':'VALIDATED','implementation_scope':'release regression and synthetic tests; not a runtime test execution','detector_fusion':'NOT_EVALUATED',
        'contact':contact['status'],'longitudinal_geometry':'AVAILABLE_UNVALIDATED' if usable else 'MISSING','research_accuracy_frozen':False}
    return result


def refine_contact_file(state_json,stage1_root,stage3_state,stage4_handoff,output_dir):
    from .camera import camera_state_for_frame, optimized_camera_dir
    import hashlib
    from .version import runtime_provenance
    from .visualization import render_selected_frame
    paths=[Path(p).resolve() for p in (state_json,stage3_state)]
    source,poses=[json.loads(p.read_text(encoding='utf-8')) for p in paths]
    handoff=None
    if stage4_handoff is not None:
        paths.append(Path(stage4_handoff).resolve())
        handoff=json.loads(paths[-1].read_text(encoding='utf-8'))
    if source.get('localization_mode')!='hybrid-3d': raise ValueError('Use a hybrid-3d baseline cache (not a previously contact-refined state)')
    camera=camera_state_for_frame(stage1_root,source['replay_context']['selected_frame'])
    result=apply_contact(source,poses,handoff,camera)
    out=Path(output_dir).resolve(); out.mkdir(parents=True,exist_ok=True)
    if out==paths[0].parent: raise ValueError('Output must differ from baseline directory')
    result['diagnostics']['runtime_provenance']=runtime_provenance()
    result['diagnostics']['contact_inputs']=[str(p) for p in paths]
    camera_path=optimized_camera_dir(stage1_root)/f"camera_state_{camera.frame_index:08d}.json"
    result['diagnostics']['contact_camera_source']={'path':str(camera_path),'sha256':hashlib.sha256(camera_path.read_bytes()).hexdigest()}
    selected=result['selected_frame_ball']
    hand={'schema_version':'stage6-downstream-handoff-1.0','selected_frame':selected['frame_index'],
          'stage7':selected['contact'],'stage8':selected['localization'],'research_accuracy_frozen':False}
    result['artifacts']={'baseline_artifacts':source.get('artifacts',{}),'ball_trajectory_state':str(out/'ball_trajectory_state.json'),
                         'downstream_handoff':str(out/'stage6_downstream_handoff.json')}
    try:
        result['artifacts']['selected_frame_overlay']=str(render_selected_frame(source['replay_context']['video_path'],selected,out/'selected_frame_ball.png'))
        overlay=cv2.imread(result['artifacts']['selected_frame_overlay'])
        if overlay is not None:
            c=selected['contact']
            label=f"Contact: {c['status']} | toucher: {c.get('track_id') or 'UNASSIGNED'}"
            cv2.putText(overlay,label,(24,90),cv2.FONT_HERSHEY_SIMPLEX,.65,(0,220,255),2)
            if c.get('image_distance_px') is not None:
                label=f"Nearest pose: {c.get('nearest_track_id') or c.get('track_id')} {c['region']} {c['image_distance_px']:.1f}px (gate {c['threshold_px']:.1f}px)"
                cv2.putText(overlay,label,(24,120),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,220,255),2)
            cv2.imwrite(result['artifacts']['selected_frame_overlay'],overlay)
    except Exception as exc: result['diagnostics']['contact_overlay_warning']=str(exc)
    c=selected['contact']; loc=selected['localization']
    report=(f"# Stage 6 v0.5.1 cached contact refinement\n\n"
            f"Frame: {selected['frame_index']}\n\nContact status: {c['status']}\n\n"
            f"Assigned toucher: {c.get('track_id')}\n\nNearest region: {c.get('region')}\n\n"
            f"Distance (px): {c.get('image_distance_px')}\n\nMethod: {selected['method']}\n\n"
            f"XYZ (m): {selected['center_xyz_world_m']}\n\nDelta X vs hybrid (m): {selected['delta_x_vs_hybrid_m']}\n\n"
            f"Fallback used: {selected['fallback']['used']}\n\n"
            "No detector inference/tracking rerun. No contact or 3D GT: accuracy unvalidated; research not frozen.\n")
    (out/'contact_refinement_report.md').write_text(report,encoding='utf-8')
    result['artifacts']['report']=str(out/'contact_refinement_report.md')
    for name,payload in [('ball_trajectory_state.json',result),('stage6_downstream_handoff.json',hand)]:
        (out/name).write_text(json.dumps(payload,indent=2,allow_nan=False),encoding='utf-8')
    return result
