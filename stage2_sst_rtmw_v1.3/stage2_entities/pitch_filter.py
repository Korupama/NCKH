"""Strict pitch membership before tracks enter pose/team/context inference."""
from copy import deepcopy

import numpy as np


def filter_tracks_to_pitch(tracks, camera):
    """Keep only forward ground intersections inside the calibrated rectangle.

    Use the bottom centre of a person's box (ground contact proxy), never the
    head/box centre or an elevated 3D joint. Balls use their centre, matching the
    Stage-6 strict-ground policy. All roles obey the same boundary, including GK.
    Unverifiable observations are excluded; original IDs are never renumbered.
    This is intended for the selected-frame pipeline, with a same-frame camera.
    """
    if camera is None or str(camera.status).upper() not in {'VALID', 'DEGRADED'}:
        raise ValueError('Pitch track filtering requires a usable Stage-1 camera')
    length = float(camera.pitch.get('length_m', 105.0))
    width = float(camera.pitch.get('width_m', 68.0))
    if not np.isfinite([length, width]).all() or min(length, width) <= 0:
        raise ValueError('Invalid pitch dimensions for track filtering')
    kept, excluded = [], []
    for original in tracks:
        track = deepcopy(original)
        observations = []
        for obs in track.get('observations', []):
            reason, anchor, xyz = None, None, None
            if obs.get('frame_index') != camera.frame_index:
                raise ValueError('Pitch track filter camera/frame mismatch')
            box = np.asarray(obs.get('bbox_xyxy', []), dtype=float)
            if box.shape != (4,) or not np.isfinite(box).all() or box[2] <= box[0] or box[3] <= box[1]:
                reason = 'INVALID_BOUNDING_BOX'
            else:
                is_ball = str(track.get('role', '')).lower() == 'ball'
                anchor = [float((box[0] + box[2]) / 2),
                          float((box[1] + box[3]) / 2 if is_ball else box[3])]
                hit = camera.intersect_z_plane(np.asarray(anchor), 0.0)[0]
                if not np.isfinite(hit).all():
                    reason = 'NO_FORWARD_PITCH_INTERSECTION'
                else:
                    xyz = hit.tolist()
                    if abs(hit[0]) > length / 2 + 1e-8 or abs(hit[1]) > width / 2 + 1e-8:
                        reason = 'OUTSIDE_PITCH'
            evidence = {'anchor_uv': anchor, 'ground_xyz_m': xyz}
            if reason:
                excluded.append({'track_id': track['track_id'], 'role': track.get('role'),
                                 'frame_index': obs.get('frame_index'), 'reason': reason,
                                 'bbox_xyxy': obs.get('bbox_xyxy'), **evidence})
            else:
                obs['pitch_membership'] = {'status': 'INSIDE', **evidence}
                observations.append(obs)
        if observations:
            track['observations'] = observations
            kept.append(track)
    report = {
        'policy': 'STRICT_CALIBRATED_PITCH', 'margin_m': 0.0,
        'frame_index': camera.frame_index, 'camera_status': camera.status,
        'pitch_length_m': length, 'pitch_width_m': width,
        'person_anchor': 'BBOX_BOTTOM_CENTER', 'ball_anchor': 'BBOX_CENTER',
        'input_track_count': len(tracks), 'kept_track_count': len(kept),
        'excluded_track_count': len(tracks) - len(kept),
        'excluded': excluded,
    }
    return kept, report
