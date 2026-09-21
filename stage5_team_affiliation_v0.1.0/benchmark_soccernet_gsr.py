from __future__ import annotations
import argparse, json
from pathlib import Path

from stage5_team_affiliation.gsr_benchmark import SoccerNetGSRDataset, run_benchmark


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 5 third-party benchmark on SoccerNet-GSR")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("inspect", help="Inspect SoccerNet-GSR split and verify local image/video resolution")
    p.add_argument("--gsr-root", required=True)
    p.add_argument("--split", default="valid")
    p.add_argument("--max-sequences", type=int, default=5)

    p = sub.add_parser("run", help="Run track-level team-affiliation benchmark")
    p.add_argument("--gsr-root", required=True)
    p.add_argument("--split", default="valid")
    p.add_argument("--method", choices=["bbox-color", "stage5-color", "external", "residual-v1", "residual-v2", "residual-v3"], default="bbox-color")
    p.add_argument("--residual-config", help="Frozen ResidualConfig JSON; tune on TRAIN only")
    p.add_argument("--baseline-summary", help="Paired V0 bbox-color summary with sequence_metrics.jsonl")
    p.add_argument("--pose-cache-root", default=None, help="Optional per-sequence pose cache root for stage5-color")
    p.add_argument("--external-predictions-root", default=None, help="Required for method=external")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--progress-every", type=int, default=5)

    p = sub.add_parser(
        "evaluate-residual-heldout",
        help="Run leakage-safe full VALID evaluation using a TRAIN-calibrated residual-v3 config",
    )
    p.add_argument("--gsr-root", required=True)
    p.add_argument("--split", choices=["valid"], default="valid")
    p.add_argument("--residual-config", required=True)
    p.add_argument("--calibration-report", required=True)
    p.add_argument("--baseline-summary", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--progress-every", type=int, default=1)

    p = sub.add_parser(
        "reevaluate-residual-cache",
        help="Reapply a TRAIN-calibrated residual policy to a completed VALID cache without reading images",
    )
    p.add_argument("--gsr-root", required=True)
    p.add_argument("--split", choices=["valid"], default="valid")
    p.add_argument("--source-dir", required=True)
    p.add_argument("--residual-config", required=True)
    p.add_argument("--calibration-report", required=True)
    p.add_argument("--baseline-summary", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--progress-every", type=int, default=10)

    p = sub.add_parser("compare", help="Compare multiple benchmark_summary.json files")
    p.add_argument("--summary", action="append", required=True, help="Path to benchmark_summary.json; repeat for each method")
    p.add_argument("--output", default=None, help="Optional markdown output path")

    p = sub.add_parser("calibrate-residual", help="TRAIN-only calibration from a completed residual-v3 run")
    p.add_argument("--gsr-root", required=True)
    p.add_argument("--split", default="train")
    p.add_argument("--prediction-dir", required=True, help="Completed abstaining residual-v3 TRAIN output")
    p.add_argument("--output-dir", required=True)

    p = sub.add_parser(
        "audit-role-components",
        help="Audit referee/GK-role/GK-team metrics from stored predictions without reading images",
    )
    p.add_argument("--gsr-root", required=True)
    p.add_argument("--split", choices=["train", "valid"], default="valid")
    p.add_argument("--prediction-dir", required=True)
    p.add_argument("--output-dir", required=True)

    args = ap.parse_args()
    if args.cmd == "inspect":
        d = SoccerNetGSRDataset(args.gsr_root, args.split).inspect(args.max_sequences)
        print(json.dumps(d, indent=2, ensure_ascii=False))
        return
    if args.cmd == "compare":
        rows=[]
        for raw in args.summary:
            d=json.loads(Path(raw).read_text(encoding="utf-8"))
            g=(d.get("groups") or {}).get("all_team_tracks") or {}
            o=(d.get("groups") or {}).get("outfield") or {}
            k=(d.get("groups") or {}).get("goalkeeper") or {}
            rows.append({
                "method":d.get("method"),"split":d.get("split"),"sequences":d.get("num_sequences"),
                "all_overall":g.get("micro_overall_accuracy"),"all_coverage":g.get("micro_coverage"),
                "all_selective":g.get("micro_selective_accuracy"),"outfield_overall":o.get("micro_overall_accuracy"),
                "gk_overall":k.get("micro_overall_accuracy"),"macro_f1":g.get("macro_f1"),
                "ARI":g.get("macro_ARI"),"NMI":g.get("macro_NMI"),
                "ref_contam":(d.get("referee") or {}).get("team_contamination_rate"),
            })
        pct=lambda v: "—" if v is None else f"{100*float(v):.2f}%"
        num=lambda v: "—" if v is None else f"{float(v):.4f}"
        lines=["# Stage 5 benchmark comparison","",
               "| Method | Split | Seq | Overall acc | Coverage | Selective acc | Outfield acc | GK acc | Macro-F1 | ARI | NMI | Ref contamination |",
               "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for r in rows:
            lines.append(f"| {r['method']} | {r['split']} | {r['sequences']} | {pct(r['all_overall'])} | {pct(r['all_coverage'])} | {pct(r['all_selective'])} | {pct(r['outfield_overall'])} | {pct(r['gk_overall'])} | {num(r['macro_f1'])} | {num(r['ARI'])} | {num(r['NMI'])} | {pct(r['ref_contam'])} |")
        text="\n".join(lines)+"\n"
        print(text)
        if args.output:
            Path(args.output).write_text(text,encoding="utf-8")
        return
    if args.cmd == "calibrate-residual":
        from stage5_team_affiliation.residual_calibration import calibrate_residual_train
        report = calibrate_residual_train(dataset_root=args.gsr_root, split=args.split,
                                          prediction_dir=args.prediction_dir,
                                          output_dir=args.output_dir)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return
    if args.cmd == "audit-role-components":
        from stage5_team_affiliation.role_component_audit_v032 import (
            audit_role_components,
        )
        report = audit_role_components(
            dataset_root=args.gsr_root,
            split=args.split,
            prediction_dir=args.prediction_dir,
            output_dir=args.output_dir,
        )
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return
    if args.cmd == "evaluate-residual-heldout":
        from stage5_team_affiliation.heldout_v027 import load_heldout_candidate
        from stage5_team_affiliation.residual_benchmark import run_residual_benchmark
        cfg, provenance = load_heldout_candidate(
            config_path=args.residual_config,
            calibration_report_path=args.calibration_report,
            dataset_root=args.gsr_root,
            evaluation_split=args.split,
        )
        summary = run_residual_benchmark(
            dataset_root=args.gsr_root,
            split=args.split,
            output_dir=args.output_dir,
            variant="V3",
            progress_every=args.progress_every,
            residual_config=cfg,
            baseline_summary=args.baseline_summary,
            heldout_evaluation=True,
            calibration_provenance=provenance,
        )
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return
    if args.cmd == "reevaluate-residual-cache":
        from stage5_team_affiliation.heldout_v027 import load_heldout_candidate
        from stage5_team_affiliation.residual_cache_v028 import reevaluate_residual_cache
        cfg, provenance = load_heldout_candidate(
            config_path=args.residual_config,
            calibration_report_path=args.calibration_report,
            dataset_root=args.gsr_root,
            evaluation_split=args.split,
        )
        summary = reevaluate_residual_cache(
            dataset_root=args.gsr_root,
            split=args.split,
            source_dir=args.source_dir,
            output_dir=args.output_dir,
            residual_config=cfg,
            baseline_summary=args.baseline_summary,
            calibration_provenance=provenance,
            progress_every=args.progress_every,
        )
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return
    if args.method.startswith("residual-"):
        if args.pose_cache_root or args.external_predictions_root:
            ap.error("Residual oracle experiment uses GT human bbox crops, not pose/external predictions")
        from stage5_team_affiliation.residual_benchmark import run_residual_benchmark
        from stage5_team_affiliation.residual_config import ResidualConfig
        cfg = ResidualConfig(**json.loads(Path(args.residual_config).read_text(encoding="utf-8"))) if args.residual_config else ResidualConfig()
        s = run_residual_benchmark(dataset_root=args.gsr_root, split=args.split, output_dir=args.output_dir,
                                  variant=args.method[-2:].upper(), limit=args.limit, progress_every=args.progress_every,
                                  residual_config=cfg, baseline_summary=args.baseline_summary)
        print(json.dumps(s, indent=2, ensure_ascii=False))
        return
    if args.residual_config or args.baseline_summary:
        ap.error("--residual-config/--baseline-summary apply only to residual-v1/v2/v3")
    s = run_benchmark(
        dataset_root=args.gsr_root, split=args.split, output_dir=args.output_dir,
        method=args.method, pose_cache_root=args.pose_cache_root,
        external_predictions_root=args.external_predictions_root,
        limit=args.limit, progress_every=args.progress_every,
    )
    print(json.dumps(s, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
