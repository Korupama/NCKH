"""Deduplicate proposals without averaging away optimized YOLO diameters."""
from copy import deepcopy


def prefer_primary_candidates(primary, auxiliary):
    """Do not compare uncalibrated scores from two different detectors.

    The dedicated ball detector supplies candidates when present. SST only
    fills frames with no primary detection; all decisions remain auditable.
    """
    result = {}
    fallback_frames = []
    for fi in sorted(set(primary) | set(auxiliary)):
        for candidate in list(primary.get(fi, [])) + list(auxiliary.get(fi, [])):
            if candidate.frame_index != fi or candidate.coordinate_space != 'RAW_DISTORTED_PIXEL':
                raise ValueError('Candidate frame/pixel-space mismatch')
        rows = primary.get(fi, [])
        if not rows:
            rows = auxiliary.get(fi, [])
            if rows:
                fallback_frames.append(fi)
        result[fi] = deepcopy(rows)
    return result, {'policy': 'DEDICATED_BALL_FIRST_SST_IF_MISSING',
                    'fallback_frames': fallback_frames,
                    'primary_count': sum(map(len, primary.values())),
                    'auxiliary_count': sum(map(len, auxiliary.values()))}

def _iou(a,b):
    w=max(0,min(a[2],b[2])-max(a[0],b[0])); h=max(0,min(a[3],b[3])-max(a[1],b[1]))
    union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-w*h
    return w*h/union if union>0 else 0

def fuse_candidates(primary,auxiliary,iou_threshold=.5):
    result={}; counts={'primary':0,'auxiliary':0,'duplicates':0,'output':0}
    for fi in sorted(set(primary)|set(auxiliary)):
        rows=[]
        for c in list(primary.get(fi,[]))+list(auxiliary.get(fi,[])):
            if c.frame_index!=fi or c.coordinate_space!='RAW_DISTORTED_PIXEL':
                raise ValueError('Candidate frame/pixel-space mismatch')
            match=next((r for r in rows if _iou(r.bbox_xyxy,c.bbox_xyxy)>=iou_threshold),None)
            if match is None:
                match=deepcopy(c); match.metadata['sources']=[c.source]
                match.metadata['source_candidate_ids']=[c.candidate_id]; rows.append(match)
            else:
                counts['duplicates']+=1
                match.metadata['sources']=sorted(set(match.metadata['sources']+[c.source]))
                match.metadata['source_candidate_ids'].append(c.candidate_id)
        result[fi]=rows
        counts['primary']+=len(primary.get(fi,[])); counts['auxiliary']+=len(auxiliary.get(fi,[])); counts['output']+=len(rows)
    return result,counts
