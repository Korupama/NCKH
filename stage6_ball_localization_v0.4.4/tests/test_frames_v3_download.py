from pathlib import Path

from ball_localization.datasets.soccernet_frames_v3 import (
    build_frames_v3_download_plan,
    summarize_frames_v3_plan,
)

HEADER='league,season,match,main_action,action,replay,img_w,img_h,is_ball,ball_bbox,calibration,JaC@0.005,JaC@0.01,JaC@0.02,ball_3D,rep_error,optimized_error,optimized_d,set\n'


def test_frames_v3_plan_matches_huggingface_tree_and_deduplicates(tmp_path: Path):
    csv_path = tmp_path / 'SNv3D.csv'
    rows = [
        'england_epl,2014-2015,2015-02-21 - 18-00 Chelsea 1 - 1 Burnley,3,True,nan,1920,1080,True,"[1,2,8,8]",{},0,0,0,"[1 2 -0.11]",0,0,8,test',
        'england_epl,2014-2015,2015-02-21 - 18-00 Chelsea 1 - 1 Burnley,3,False,1,1920,1080,True,"[1,2,8,8]",{},0,0,0,"[1 2 -0.11]",0,0,8,test',
        'italy_serie-a,2016-2017,2017-05-28 - 19-00 Sampdoria 2 - 4 Napoli,3,False,1,1920,1080,True,"[1,2,8,8]",{},0,0,0,"[1 2 -0.11]",0,0,8,test',
    ]
    csv_path.write_text(HEADER + '\n'.join(rows) + '\n', encoding='utf-8')
    plan = build_frames_v3_download_plan(csv_path, split='test')
    assert len(plan) == 2
    assert plan[0].remote_path == (
        'england_epl/2014-2015/2015-02-21 - 18-00 Chelsea 1 - 1 Burnley/Frames-v3.zip'
    )
    assert plan[1].remote_path == (
        'italy_serie-a/2016-2017/2017-05-28 - 19-00 Sampdoria 2 - 4 Napoli/Frames-v3.zip'
    )


def test_frames_v3_plan_reports_existing_archive(tmp_path: Path):
    csv_path = tmp_path / 'SNv3D.csv'
    row = 'england_epl,2014-2015,2015-02-21 - 18-00 Chelsea 1 - 1 Burnley,3,True,nan,1920,1080,True,"[1,2,8,8]",{},0,0,0,"[1 2 -0.11]",0,0,8,test\n'
    csv_path.write_text(HEADER + row, encoding='utf-8')
    root = tmp_path / 'SoccerNet'
    archive = root / 'england_epl' / '2014-2015' / '2015-02-21 - 18-00 Chelsea 1 - 1 Burnley' / 'Frames-v3.zip'
    archive.parent.mkdir(parents=True)
    archive.write_bytes(b'placeholder')
    summary = summarize_frames_v3_plan(csv_path, root, split='test')
    assert summary['unique_match_archives'] == 1
    assert summary['already_present'] == 1
    assert summary['missing'] == 0
