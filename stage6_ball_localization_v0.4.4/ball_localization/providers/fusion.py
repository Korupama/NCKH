"""Deduplicate proposals without averaging away optimized YOLO diameters."""
from copy import deepcopy

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
