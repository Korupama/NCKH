from pathlib import Path
import zipfile

import cv2
import numpy as np
from ball_localization.datasets import SoccerNetV3DCSV, SoccerNetImageResolver, read_image_reference
from ball_localization.datasets.soccernet_v3d import parse_xyz

HEADER='league,season,match,main_action,action,replay,img_w,img_h,is_ball,ball_bbox,calibration,JaC@0.005,JaC@0.01,JaC@0.02,ball_3D,rep_error,optimized_error,optimized_d,set\n'

def test_csv_adapter_canonicalizes_soccernet_world(tmp_path:Path):
    p=tmp_path/'x.csv'
    p.write_text(HEADER+'L,2024,M,3,True,nan,1920,1080,True,"[1, 2, 11, 12]",{},0,0,0,"[1.0, 2.0, -0.11]",0,0,10,test\n',encoding='utf-8')
    ds=SoccerNetV3DCSV(p); r=ds.records[0]
    assert r.basename=='3.png'
    assert r.ball_bbox==[1,2,12,14]
    assert r.ball_3d_soccernet==[1.0,2.0,-0.11]
    assert r.ball_3d==[1.0,-2.0,0.11]
    summary=ds.summary()['sets']['test']
    assert summary['ball_rows']==1
    assert summary['source_negative_z_rows']==1
    assert summary['canonical_nonnegative_z_rows']==1

def test_raw_xyz_parser_handles_numpy_space_separated():
    assert parse_xyz('[12.5 -4.25 -1.125]') == [12.5,-4.25,-1.125]

def test_csv_adapter_parses_numpy_space_separated_ball3d(tmp_path:Path):
    p=tmp_path/'numpy_style.csv'
    p.write_text(HEADER+'L,2024,M,3,True,nan,1920,1080,True,"[1, 2, 11, 12]",{},0,0,0,"[12.5 -4.25 -1.125]",0,0,10,test\n',encoding='utf-8')
    ds=SoccerNetV3DCSV(p)
    r=ds.records[0]
    assert r.ball_3d_soccernet == [12.5,-4.25,-1.125]
    assert r.ball_3d == [12.5,4.25,1.125]
    summary=ds.summary()['sets']['test']
    assert summary['ball_3d_rows']==1
    assert summary['ball_3d_parse_failures']==0


def test_csv_adapter_parses_numpy_array_wrapper_ball3d(tmp_path:Path):
    p=tmp_path/'array_style.csv'
    p.write_text(HEADER+'L,2024,M,3,True,nan,1920,1080,True,"[1, 2, 11, 12]",{},0,0,0,"array([12.5, -4.25, -1.125])",0,0,10,test\n',encoding='utf-8')
    ds=SoccerNetV3DCSV(p)
    assert ds.records[0].ball_3d == [12.5,4.25,1.125]


def _single_record_csv(tmp_path:Path, *, replay:bool=True)->Path:
    p=tmp_path/'resolver.csv'
    action='False' if replay else 'True'
    replay_id='1' if replay else 'nan'
    p.write_text(HEADER+f'L,2024,M,3,{action},{replay_id},64,48,True,"[10, 10, 8, 8]",{{}},0,0,0,"[1.0 2.0 -0.11]",0,0,8,test\n',encoding='utf-8')
    return p


def test_image_resolver_reads_frames_v3_zip_without_extraction(tmp_path:Path):
    csv_path=_single_record_csv(tmp_path,replay=True)
    record=SoccerNetV3DCSV(csv_path).records[0]
    match_dir=tmp_path/'L'/'2024'/'M'
    match_dir.mkdir(parents=True)
    image=np.full((48,64,3),127,dtype=np.uint8)
    ok,encoded=cv2.imencode('.png',image)
    assert ok
    archive=match_dir/'Frames-v3.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr('Frames-v3/3_1.png',encoded.tobytes())
    resolver=SoccerNetImageResolver(tmp_path)
    decoded,source=resolver.read(record)
    assert source is not None
    assert source.kind=='frames-v3-zip'
    assert source.member=='Frames-v3/3_1.png'
    assert source.reference.startswith('zip://')
    assert decoded is not None and decoded.shape==(48,64,3)
    assert int(decoded[0,0,0])==127
    reread=read_image_reference(source.reference)
    assert reread is not None and reread.shape==decoded.shape


def test_image_resolver_prefers_loose_file_over_zip(tmp_path:Path):
    csv_path=_single_record_csv(tmp_path,replay=False)
    record=SoccerNetV3DCSV(csv_path).records[0]
    match_dir=tmp_path/'L'/'2024'/'M'
    match_dir.mkdir(parents=True)
    loose=np.full((48,64,3),33,dtype=np.uint8)
    assert cv2.imwrite(str(match_dir/'3.png'),loose)
    ok,encoded=cv2.imencode('.png',np.full((48,64,3),200,dtype=np.uint8))
    assert ok
    with zipfile.ZipFile(match_dir/'Frames-v3.zip','w') as zf:
        zf.writestr('3.png',encoded.tobytes())
    resolver=SoccerNetImageResolver(tmp_path)
    decoded,source=resolver.read(record)
    assert source is not None and source.kind=='filesystem'
    assert decoded is not None and int(decoded[0,0,0])==33


def test_csv_adapter_handles_numeric_action_flag_and_blank_replay(tmp_path:Path):
    p=tmp_path/'numeric_action.csv'
    rows=[
        'L,2024,M,14,1.0,,1920,1080,True,"[10, 10, 8, 8]",{},0,0,0,"[1.0 2.0 -0.11]",0,0,8,test',
        'L,2024,M,14,0.0,2.0,1920,1080,True,"[10, 10, 8, 8]",{},0,0,0,"[1.0 2.0 -0.11]",0,0,8,test',
    ]
    p.write_text(HEADER+'\n'.join(rows)+'\n',encoding='utf-8')
    ds=SoccerNetV3DCSV(p)
    assert ds.records[0].action is True
    assert ds.records[0].is_action_frame is True
    assert ds.records[0].basename=='14.png'
    assert ds.records[1].action is False
    assert ds.records[1].is_action_frame is False
    assert ds.records[1].basename=='14_2.png'


def test_csv_adapter_defensively_infers_action_from_missing_replay(tmp_path:Path):
    p=tmp_path/'blank_replay.csv'
    p.write_text(HEADER+'L,2024,M,14,0.0,,1920,1080,True,"[10, 10, 8, 8]",{},0,0,0,"[1.0 2.0 -0.11]",0,0,8,test\n',encoding='utf-8')
    ds=SoccerNetV3DCSV(p)
    r=ds.records[0]
    assert r.action is False
    assert r.is_action_frame is True
    assert r.basename=='14.png'
    summary=ds.summary()['sets']['test']
    assert summary['action_frame_rows']==1
    assert summary['action_inferred_from_missing_replay_rows']==1


def test_image_resolver_matches_numeric_action_export_inside_frames_v3_zip(tmp_path:Path):
    p=tmp_path/'numeric_resolver.csv'
    rows=[
        'L,2024,M,14,1.0,,64,48,True,"[10, 10, 8, 8]",{},0,0,0,"[1.0 2.0 -0.11]",0,0,8,test',
        'L,2024,M,14,0.0,2.0,64,48,True,"[10, 10, 8, 8]",{},0,0,0,"[1.0 2.0 -0.11]",0,0,8,test',
    ]
    p.write_text(HEADER+'\n'.join(rows)+'\n',encoding='utf-8')
    records=SoccerNetV3DCSV(p).records
    match_dir=tmp_path/'L'/'2024'/'M'
    match_dir.mkdir(parents=True)
    image=np.full((48,64,3),101,dtype=np.uint8)
    ok,encoded=cv2.imencode('.png',image)
    assert ok
    with zipfile.ZipFile(match_dir/'Frames-v3.zip','w',compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr('14.png',encoded.tobytes())
        zf.writestr('Frames-v3/14_2.png',encoded.tobytes())
    resolver=SoccerNetImageResolver(tmp_path)
    for expected,record in zip(('14.png','14_2.png'),records):
        decoded,source=resolver.read(record)
        assert record.basename==expected
        assert source is not None and source.kind=='frames-v3-zip'
        assert decoded is not None and decoded.shape==(48,64,3)
