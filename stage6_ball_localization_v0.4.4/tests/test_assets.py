from ball_localization.assets import ASSETS

def test_official_assets_registry():
    assert ASSETS['yolo-sn-ball-opt']['filename']=='yolo-sn-ball-opt.pt'
    assert ASSETS['snv3d-csv']['url'].endswith('/SNv3D.csv')
