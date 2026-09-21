from ball_localization.evaluation import summarize_2d

def test_metrics_perfect():
    rows=[{'gt_bbox':[0,0,10,10],'predictions':[{'bbox_xyxy':[0,0,10,10],'score':.9}],'img_w':100,'img_h':100}]
    m=summarize_2d(rows,candidate_k=5); assert m['precision']==1 and m['recall']==1 and m['AP50']>0.999 and m['mAP50_95']>0.999 and m['CandidateRecall@5']==1

def test_candidate_recall_can_exceed_top1_recall():
    rows=[{'gt_bbox':[0,0,10,10],'predictions':[{'bbox_xyxy':[50,50,60,60],'score':.9},{'bbox_xyxy':[0,0,10,10],'score':.8}],'img_w':100,'img_h':100}]
    m=summarize_2d(rows,candidate_k=2); assert m['recall']==0 and m['CandidateRecall@2']==1


def test_candidate_recall_multiple_k_values():
    from ball_localization.evaluation.metrics2d import summarize_2d
    rows=[{
        "img_w":100,"img_h":100,"gt_bbox":[40,40,50,50],
        "predictions":[
            {"bbox_xyxy":[0,0,10,10],"score":0.9},
            {"bbox_xyxy":[40,40,50,50],"score":0.8},
        ],
    }]
    m=summarize_2d(rows,candidate_k=5,candidate_ks=[1,5,10])
    assert m["CandidateRecall@1"] == 0.0
    assert m["CandidateRecall@5"] == 1.0
    assert m["CandidateRecall@10"] == 1.0
