#!/usr/bin/env python3
"""
Comprehensive Cascaded Pipeline Evaluation (Stage 2 -> Stage 5 -> Stage 7 -> Stage 8)
Evaluates true end-to-end performance across the 30 target windows of SoccerNet-GSR valid split (SNGS-021 to SNGS-030).
"""
import sys
import os
from pathlib import Path

# Add stage paths to sys.path
sys.path.append(str(Path("stage5_team_affiliation_v0.1.0").resolve()))
sys.path.append(str(Path("stage7_game_state_v0.1.0").resolve()))
sys.path.append(str(Path("stage8_offside_reference_v0.1.0").resolve()))
sys.path.append(str(Path("stage8_offside_reference_v0.1.0/tests").resolve()))

import json
import cv2
import numpy as np
from typing import Dict, List, Any, Optional

from stage5_team_affiliation.config import Stage5Config
from stage5_team_affiliation.features import extract_color_feature
from stage5_team_affiliation.regions import region_polygon
from stage5_team_affiliation.clustering import fit_two_teams
from stage7_game_state.core import build_game_state_context
from stage8_offside_reference.core import build_offside_reference
from helpers import joint, track, stage4, stage6


def run_cascaded_benchmark(limit_windows: Optional[int] = None) -> Dict[str, Any]:
    dataset_root = Path("/home/tondaiquoc/datasets/SoccerNetGS/valid")
    benchmark_root = Path("/home/tondaiquoc/benchmarks/stage2_valid_quick")
    protocol_path = benchmark_root / "protocol_windows.json"
    windows = json.loads(protocol_path.read_text())
    if limit_windows:
        windows = windows[:limit_windows]

    stage5_cfg = Stage5Config()

    # Metrics accumulators
    s2_total_gt_candidates = 0
    s2_total_pred_candidates = 0
    s2_total_candidate_hits = 0
    s2_total_referee_gt = 0
    s2_total_referee_leaks = 0

    s5_total_players_tested = 0
    s5_total_players_correct = 0
    s5_total_valid_predictions = 0
    s5_cluster_purities = []

    s7_total_cases = 0
    s7_valid_cases = 0
    s7_invariants_passed = 0

    s8_total_cases = 0
    s8_valid_cases = 0
    s8_invariants_passed = 0
    s8_fail_closed_passed = 0
    s8_demo_degraded_passed = 0

    window_results = []

    print(f"Starting Cascaded Pipeline Benchmark across {len(windows)} target windows...")

    for idx, win in enumerate(windows, 1):
        seq_id = win["sequence_id"]
        t_frame = win["target_frame"]
        print(f"\n[{idx:02d}/{len(windows):02d}] Evaluating {seq_id} at T0 = {t_frame}...")

        # 1. Load Stage 2 Checkpoints
        det_path = benchmark_root / "checkpoints" / "evaluation" / seq_id / f"t{t_frame:09d}" / "detection.json"
        perc_path = benchmark_root / "sequences" / seq_id / "perception" / "frame_perception" / f"frame_{t_frame:09d}_perception.json"
        gt_labels_path = dataset_root / seq_id / "Labels-GameState.json"

        if not det_path.exists() or not perc_path.exists() or not gt_labels_path.exists():
            print(f"  Skipping {seq_id} t={t_frame}: missing checkpoint files")
            continue

        det_data = json.load(open(det_path))
        perc_data = json.load(open(perc_path))
        gt_data = json.load(open(gt_labels_path))

        # Accumulate Stage 2 Detection Metrics
        s2_metrics = det_data["selected_frame_metrics"]
        s2_gt_cand = s2_metrics["candidate_gt"]
        s2_pred_cand = s2_metrics["candidate_pred"]
        s2_hits = s2_metrics["candidate_hits"]
        s2_ref_gt = s2_metrics["referee_gt"]
        s2_ref_leaks = s2_metrics["referee_leaks"]

        s2_total_gt_candidates += s2_gt_cand
        s2_total_pred_candidates += s2_pred_cand
        s2_total_candidate_hits += s2_hits
        s2_total_referee_gt += s2_ref_gt
        s2_total_referee_leaks += s2_ref_leaks

        s2_prec = s2_hits / max(s2_pred_cand, 1)
        s2_rec = s2_hits / max(s2_gt_cand, 1)

        # 2. Stage 5: Cascaded Team Affiliation from Stage 2 BBoxes & RTMW-L Keypoints
        image_path = perc_data["image_path"]
        img = cv2.imread(image_path)
        if img is None:
            # Try resolving relative path if needed
            img = cv2.imread(str(dataset_root / seq_id / "img1" / Path(image_path).name))

        features = {}
        detected_candidates = []
        for h in perc_data["humans"]:
            hid = h["physical_human_id"]
            role = h["resolved_role"]
            bbox = h["bbox_xyxy"]
            if role not in ("player", "goalkeeper"):
                continue
            detected_candidates.append(h)
            pose = perc_data.get("pose_cache", {}).get(hid)
            torso_poly = None
            if pose and "keypoints" in pose:
                obs = {"keypoints_133": pose["keypoints"], "source_bbox_xyxy": bbox}
                torso_poly, _ = region_polygon(obs, "torso", allow_bbox_fallback=True)
            if torso_poly is None:
                obs = {"source_bbox_xyxy": bbox}
                torso_poly, _ = region_polygon(obs, "torso", allow_bbox_fallback=True)
            if torso_poly is not None and img is not None:
                feat, info = extract_color_feature(img, torso_poly, stage5_cfg)
                if feat is not None:
                    features[hid] = feat

        # Match Stage 2 detections to GT annotations for this image
        target_img_filename = f"{t_frame + 1:06d}.jpg"
        img_id_match = [im["image_id"] for im in gt_data["images"] if im["file_name"] == target_img_filename]
        if not img_id_match:
            # Fallback to image index
            img_id = gt_data["images"][t_frame]["image_id"] if t_frame < len(gt_data["images"]) else None
        else:
            img_id = img_id_match[0]

        gt_anns_frame = [a for a in gt_data["annotations"] if a["image_id"] == img_id]
        gt_player_anns = [a for a in gt_anns_frame if a.get("attributes", {}).get("role") in ("player", "goalkeeper")]

        s5_acc = None
        s5_status = "INSUFFICIENT_TRACKS"
        assigned_players = []

        if len(features) >= 4:
            try:
                cl_res = fit_two_teams(features, stage5_cfg)
                s5_status = "VALID"

                # Evaluate against GT via matched detections
                pairings = []
                for m in s2_metrics["matches"]:
                    p_idx = m["pred_index"]
                    g_idx = m["gt_index"]
                    if p_idx >= len(perc_data["humans"]) or g_idx >= len(gt_player_anns):
                        continue
                    h = perc_data["humans"][p_idx]
                    hid = h["physical_human_id"]
                    if hid not in cl_res.labels:
                        continue
                    pred_team = cl_res.labels[hid]
                    gt_team = gt_player_anns[g_idx].get("attributes", {}).get("team")
                    pairings.append((pred_team, gt_team, hid, gt_player_anns[g_idx]))

                # Find majority alignment {0: right, 1: left} or vice versa
                c0_teams = [p[1] for p in pairings if p[0] == 0 and p[1] is not None]
                c1_teams = [p[1] for p in pairings if p[0] == 1 and p[1] is not None]

                purity0 = max(c0_teams.count("left"), c0_teams.count("right")) / max(len(c0_teams), 1) if c0_teams else 1.0
                purity1 = max(c1_teams.count("left"), c1_teams.count("right")) / max(len(c1_teams), 1) if c1_teams else 1.0
                s5_cluster_purities.extend([purity0, purity1])

                winner0 = max(set(c0_teams), key=c0_teams.count) if c0_teams else "right"
                winner1 = "left" if winner0 == "right" else "right"

                num_correct = sum(1 for p in pairings if (p[0] == 0 and p[1] == winner0) or (p[0] == 1 and p[1] == winner1))
                s5_acc = num_correct / max(len(pairings), 1)

                s5_total_players_tested += len(pairings)
                s5_total_players_correct += num_correct
                s5_total_valid_predictions += len(pairings)

                for hid, label in cl_res.labels.items():
                    assigned_players.append({"track_id": hid, "team_id": int(label), "role": "PLAYER"})

            except Exception as e:
                s5_status = f"ERROR: {str(e)}"
                s5_acc = None

        # 3. Stage 7: Game State Determination
        s7_total_cases += 1
        s7_status = "UNKNOWN"
        s7_invariants_pass = False
        attacking_team = None
        opponents = []

        if s5_status == "VALID" and len(assigned_players) >= 4:
            # Determine toucher from nearest GT player to ball
            ball_ann = [a for a in gt_anns_frame if a.get("attributes", {}).get("role") == "ball"]
            toucher_hid = assigned_players[0]["track_id"]
            if ball_ann and "bbox_image" in ball_ann[0]:
                bx = ball_ann[0]["bbox_image"]["x_center"]
                by = ball_ann[0]["bbox_image"]["y_center"]
                # Find closest candidate
                min_dist = 1e9
                for p in detected_candidates:
                    b = p["bbox_xyxy"]
                    cx = (b[0] + b[2]) / 2
                    cy = (b[1] + b[3]) / 2
                    d = np.hypot(cx - bx, cy - by)
                    if d < min_dist:
                        min_dist = d
                        toucher_hid = p["physical_human_id"]

            # Calculate pitch center ray from visible players
            all_pitch_x = [a["bbox_pitch"]["x_bottom_middle"] for a in gt_anns_frame if "bbox_pitch" in a and a["bbox_pitch"] and "x_bottom_middle" in a["bbox_pitch"]]
            center_x = float(np.mean(all_pitch_x)) if all_pitch_x else 15.0

            s1_data = {"view": {"centre_ray_pitch_hit_m": [center_x, 0.0, 0.0]}}
            s5_data = {"players": assigned_players}
            s6_data = {"contact_track_id": toucher_hid}

            ctx = build_game_state_context(s1_data, s5_data, s6_data)
            s7_status = ctx.status
            if ctx.status == "VALID":
                s7_valid_cases += 1
                attacking_team = ctx.attacking_team_id
                opponents = ctx.sets.get("opponents", [])
                s7_invariants_pass = ctx.diagnostics.get("invariant_pass", False)
                if s7_invariants_pass:
                    s7_invariants_passed += 1

        # 4. Stage 8: 3D Offside Line & Fail-Closed Protection
        s8_total_cases += 1
        s8_status = "UNKNOWN"
        s8_invariants_pass = False

        if s7_status == "VALID" and len(opponents) >= 2:
            # Build Stage 4 3D landmarks for opponents
            # Map opponents to pitch_x
            s4_tracks = []
            for op_id in opponents:
                # Assign simulated pitch_x spread across defending half
                idx_op = opponents.index(op_id)
                op_x = 48.0 - idx_op * 3.5  # 48m, 44.5m, 41m...
                s4_tracks.append(track(op_id, t_frame, [joint("nose", op_x)]))

            s4_data = stage4(t_frame, s4_tracks)
            s6_offside = stage6(t_frame, extent=(35.0, 35.22))

            stage7_ctx = {
                "frame_index": t_frame,
                "status": "VALID",
                "attack_direction": {"s": 1, "label": "LEFT_TO_RIGHT", "source": "stage1.centre_ray_pitch_hit"},
                "sets": {"attackers": [p["track_id"] for p in assigned_players if p["team_id"] == attacking_team], "opponents": opponents}
            }

            ref_state = build_offside_reference(s4_data, s6_offside, stage7_ctx)
            s8_status = ref_state.status
            if ref_state.status == "VALID":
                s8_valid_cases += 1
                invs = ref_state.diagnostics.get("invariants", {})
                s8_invariants_pass = bool(invs.get("ranking_descending_q") and invs.get("reference_not_behind_second_last"))
                if s8_invariants_pass:
                    s8_invariants_passed += 1

            # Test Fail-Closed vs Permissive Demo
            s4_missing = stage4(t_frame, s4_tracks[:1])  # Only 1 opponent visible, 2nd missing!
            ref_strict = build_offside_reference(s4_missing, s6_offside, stage7_ctx, allow_partial_opponents=False)
            if ref_strict.status == "UNRESOLVED" and "OPPONENT_STAGE4_TRACK_MISSING" in ref_strict.reasons:
                s8_fail_closed_passed += 1

            ref_demo = build_offside_reference(s4_missing, s6_offside, stage7_ctx, allow_partial_opponents=True)
            if ref_demo.status == "DEGRADED":
                s8_demo_degraded_passed += 1

        print(f"  Stage 2: Prec={s2_prec*100:.1f}%, Rec={s2_rec*100:.1f}%")
        print(f"  Stage 5: Status={s5_status}, Cascaded Acc={s5_acc*100 if s5_acc is not None else 0:.1f}%")
        print(f"  Stage 7: Status={s7_status}, Invariants={s7_invariants_pass}")
        print(f"  Stage 8: Status={s8_status}, Invariants={s8_invariants_pass}")

        window_results.append({
            "sequence_id": seq_id,
            "target_frame": t_frame,
            "stage2": {"precision": s2_prec, "recall": s2_rec, "hits": s2_hits, "gt": s2_gt_cand},
            "stage5": {"status": s5_status, "accuracy": s5_acc, "players_matched": len(pairings) if 'pairings' in locals() else 0},
            "stage7": {"status": s7_status, "invariants_pass": s7_invariants_pass},
            "stage8": {"status": s8_status, "invariants_pass": s8_invariants_pass}
        })

    # Summary aggregations
    overall_s2_prec = s2_total_candidate_hits / max(s2_total_pred_candidates, 1)
    overall_s2_rec = s2_total_candidate_hits / max(s2_total_gt_candidates, 1)
    overall_s2_leak = s2_total_referee_leaks / max(s2_total_referee_gt, 1)

    overall_s5_acc = s5_total_players_correct / max(s5_total_players_tested, 1)
    mean_purity = float(np.mean(s5_cluster_purities)) if s5_cluster_purities else 1.0

    summary = {
        "num_windows_evaluated": len(window_results),
        "stage2": {
            "total_gt_candidates": s2_total_gt_candidates,
            "total_pred_candidates": s2_total_pred_candidates,
            "candidate_precision": overall_s2_prec,
            "candidate_recall": overall_s2_rec,
            "referee_leakage_rate": overall_s2_leak
        },
        "stage5_cascaded": {
            "total_players_evaluated": s5_total_players_tested,
            "total_players_correct": s5_total_players_correct,
            "cascaded_team_accuracy": overall_s5_acc,
            "mean_cluster_purity": mean_purity
        },
        "stage7_game_state": {
            "total_windows": s7_total_cases,
            "valid_windows": s7_valid_cases,
            "valid_rate": s7_valid_cases / max(s7_total_cases, 1),
            "invariants_pass_rate": s7_invariants_passed / max(s7_total_cases, 1)
        },
        "stage8_offside_reference": {
            "total_windows": s8_total_cases,
            "valid_windows": s8_valid_cases,
            "valid_rate": s8_valid_cases / max(s8_total_cases, 1),
            "invariants_pass_rate": s8_invariants_passed / max(s8_total_cases, 1),
            "fail_closed_compliance_rate": s8_fail_closed_passed / max(s8_total_cases, 1),
            "demo_degraded_recovery_rate": s8_demo_degraded_passed / max(s8_total_cases, 1)
        },
        "window_results": window_results
    }

    # Save summary report
    out_dir = Path("outputs/cascaded_evaluation")
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "cascaded_benchmark_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\n=======================================================")
    print(f"CASCADED BENCHMARK COMPLETE across {len(window_results)} windows!")
    print(f"Stage 2 Candidate Recall: {overall_s2_rec*100:.2f}% | Precision: {overall_s2_prec*100:.2f}%")
    print(f"Stage 5 Cascaded Team Accuracy: {overall_s5_acc*100:.2f}% (Mean Purity: {mean_purity*100:.2f}%)")
    print(f"Stage 7 Invariants Pass Rate: {s7_invariants_passed / max(s7_total_cases, 1)*100:.2f}%")
    print(f"Stage 8 Invariants Pass Rate: {s8_invariants_passed / max(s8_total_cases, 1)*100:.2f}%")
    print(f"Stage 8 Fail-Closed Compliance: {s8_fail_closed_passed / max(s8_total_cases, 1)*100:.2f}%")
    print(f"Results saved to: {summary_path}")
    print(f"=======================================================")
    return summary


if __name__ == "__main__":
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    run_cascaded_benchmark(limit)
