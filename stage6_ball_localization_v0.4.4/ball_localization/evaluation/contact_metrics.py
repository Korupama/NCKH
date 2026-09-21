"""Metrics require explicit annotations; unavailable GT stays null, never PASS."""
import numpy as np

def evaluate_contact_rows(rows):
    """Rows: predicted_x, gt_x, predicted_track, gt_track, predicted_region,
    gt_region. Each row represents one annotated selected frame, metres for X.
    Missing predictions count against coverage and contact accuracy.
    """
    geometry=[r for r in rows if r.get('gt_x') is not None and np.isfinite(r['gt_x'])]
    errors=[abs(r['predicted_x']-r['gt_x']) for r in geometry
            if r.get('predicted_x') is not None and np.isfinite(r['predicted_x'])]
    contact=[r for r in rows if r.get('gt_track') is not None]
    region=[r for r in rows if r.get('gt_region') is not None]
    return {'geometry_gt_count':len(geometry),'geometry_coverage':len(errors)/len(geometry) if geometry else None,
            'BLE_X_MAE_m':float(np.mean(errors)) if errors else None,
            'BLE_X_median_m':float(np.median(errors)) if errors else None,
            'BLE_X_P90_m':float(np.percentile(errors,90)) if errors else None,
            'BLE_X_P95_m':float(np.percentile(errors,95)) if errors else None,
            'contact_track_accuracy':sum(r.get('predicted_track')==r['gt_track'] for r in contact)/len(contact) if contact else None,
            'contact_region_accuracy':sum(r.get('predicted_region')==r['gt_region'] for r in region)/len(region) if region else None,
            'research_accuracy_frozen':False}
