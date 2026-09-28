import json
from threading import Thread
from urllib.request import urlopen

import cv2
import numpy as np
import pytest

from stage9_offside_position.pipeline import build_pipeline_context
from stage9_offside_position.webapp import render_frame_bytes, serve_demo


@pytest.fixture
def artifacts(tmp_path):
    replay = dict(selected_frame=104, image_width=320, image_height=180)
    camera = dict(frame_index=104, status='DEGRADED', image=dict(width=320, height=180),
                  intrinsics={'K': [[200,0,160],[0,200,90],[0,0,1]]},
                  extrinsics={'R_world_to_camera': [[1,0,0],[0,-1,0],[0,0,-1]],
                              'camera_center_world_m': [0,0,30]})
    observation = dict(frame_index=104, bbox_xyxy=[80,40,130,150])
    pose = dict(frame_index=104, source_bbox_xyxy=[80,40,130,150], pose_status='VALID',
                keypoints_133=[dict(name='nose', x=100, y=50, state='VALID')])
    data = dict(stage1=camera,
                stage2=dict(replay_context=replay, tracks=[dict(track_id='a',observations=[observation]),
                                                          dict(track_id='old',observations=[dict(frame_index=86,bbox_xyxy=[1,1,30,40])])]),
                stage3=dict(replay_context=replay,tracks=[dict(track_id='a',observations=[pose])]),
                stage4=dict(selected_frame=104, tracks=[dict(track_id='a',selected_frame_status='MISSING',observations=[])]),
                stage5=dict(selected_frame=104,track_team={'a':dict(team_id=0,team_status='VALID',role='player')}),
                stage6=dict(replay_context=replay,selected_frame_ball=dict(frame_index=104,status='DEGRADED',candidate=dict(center_uv=[110,150]),contact=dict(track_id='a',region='FOOT'))),
                stage7=dict(frame_index=104,status='UNRESOLVED',reasons=['CENTRE_RAY_X_AMBIGUOUS'],sets={},toucher=None))
    paths = {}
    for name, value in data.items():
        paths[name] = tmp_path / (name + '.json')
        paths[name].write_text(json.dumps(value), encoding='utf-8')
    image = tmp_path/'frame.jpg'
    cv2.imwrite(str(image),np.full((180,320,3),70,np.uint8))
    return paths, str(image)


def test_no_stage8_or_classification_even_when_upstream_unresolved(artifacts, monkeypatch):
    import stage9_offside_position.webapp as webapp
    def forbidden(*args, **kwargs):
        raise AssertionError('Stage 9 classification must not run')
    monkeypatch.setattr(webapp, 'build_offside_position_state', forbidden)
    paths, image = artifacts
    # An extra Stage 8 path is ignored, never even opened.
    paths['stage8'] = 'this-file-must-not-be-opened.json'
    ctx = build_pipeline_context(paths=paths,image_path=image)
    assert ctx.state['mode'] == 'UPSTREAM_1_7'
    assert ctx.state['game_state']['status'] == 'UNRESOLVED'
    assert 'reference' not in ctx.state and 'attackers' not in ctx.state
    assert [s['id'] for s in ctx.state['stages']] == list(range(1,8))
    assert ctx.state['counts']['detected'] == 1
    row = next(r for r in ctx.state['tracks'] if r['track_id']=='old')
    assert row['bbox'] is None and not row['active']
    assert ctx.state['tracks'][0]['root_world_m'] is None
    raw = render_frame_bytes(ctx,{'overlay':['0']})
    assert render_frame_bytes(ctx,{'overlay':['1']}) == raw
    # Boxes must still be drawn when ID labels are turned off.
    assert render_frame_bytes(ctx,{'s2':['1'],'labels':['0']}) != raw
    assert render_frame_bytes(ctx,{'s2':['1'],'track':['old']}) == raw
    assert render_frame_bytes(ctx,{'s4':['1']}) == raw


@pytest.mark.parametrize('stage', ['stage1','stage2','stage3','stage4','stage5','stage6','stage7'])
def test_rejects_mixed_frames(artifacts, stage):
    paths, image = artifacts
    value = json.loads(paths[stage].read_text())
    if 'replay_context' in value:
        value['replay_context']['selected_frame'] = 86
    elif 'selected_frame' in value:
        value['selected_frame'] = 86
    else:
        value['frame_index'] = 86
    paths[stage].write_text(json.dumps(value))
    with pytest.raises(ValueError, match='frame'):
        build_pipeline_context(paths=paths,image_path=image)


def test_missing_or_resized_background_rejected(artifacts, tmp_path):
    paths, image = artifacts
    with pytest.raises(ValueError,match='source replay'):
        build_pipeline_context(paths=paths,image_path=str(tmp_path/'missing.jpg'))
    cv2.imwrite(image,np.zeros((90,160,3),np.uint8))
    with pytest.raises(ValueError,match='image size'):
        build_pipeline_context(paths=paths,image_path=image)


def test_http_serves_pipeline_ui_state_and_all_layers(artifacts):
    paths, image = artifacts
    ctx = build_pipeline_context(paths=paths,image_path=image)
    server = serve_demo(ctx,port=0)
    worker = Thread(target=server.serve_forever,daemon=True)
    worker.start()
    base = f'http://127.0.0.1:{server.server_address[1]}'
    try:
        with urlopen(base) as response:
            html = response.read().decode('utf-8')
            assert 'pipeline.js' in html and 'toggleRef' not in html
        for route in ['/api/state','/api/health','/static/pipeline.css','/static/pipeline.js']:
            with urlopen(base+route) as response:
                assert response.status == 200
        for layer in range(1,8):
            with urlopen(base+f'/api/frame.jpg?s{layer}=1') as response:
                img=cv2.imdecode(np.frombuffer(response.read(),np.uint8),cv2.IMREAD_COLOR)
                assert img.shape == (180,320,3)
    finally:
        server.shutdown()
        server.server_close()
        worker.join()
