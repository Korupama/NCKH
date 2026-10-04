from __future__ import annotations

import argparse
from pathlib import Path

from stage9_offside_position.webapp import build_demo_context, serve_demo


def main() -> int:
    p = argparse.ArgumentParser(description="Stage 9 local web demo (stdlib HTTP server)")
    p.add_argument("--stage4")
    p.add_argument("--stage7")
    p.add_argument("--stage8")
    p.add_argument("--stage9")
    p.add_argument("--stage6", help="Optional Stage 6 handoff used when Stage 8 reference is unavailable")
    p.add_argument("--stage1")
    p.add_argument("--stage2")
    p.add_argument("--stage3")
    p.add_argument("--stage5")
    p.add_argument("--pipeline", action="store_true", help="Visualize aligned Stage 1–9 artifacts")
    p.add_argument("--project", action="store_true", help="Load aligned project frame-104 artifacts in Stage 1–9 mode")
    p.add_argument("--image")
    p.add_argument("--video")
    p.add_argument("--epsilon-m", type=float, default=1e-9)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, help="Default: 8107 for Stage 1–9; 8099 for legacy offside demo")
    p.add_argument("--demo", action="store_true", help="Use bundled example artifacts")
    args = p.parse_args()
    if args.port is None:
        args.port = 8107 if args.project or args.pipeline else 8099

    root = Path(__file__).resolve().parent
    if args.project or args.pipeline:
        from stage9_offside_position.pipeline import build_pipeline_context, project_paths, rebuild_project_stage7, rebuild_project_stage8, rebuild_project_stage9
        if args.demo:
            p.error("Stage 1–9 mode does not accept --demo")
        paths = project_paths(root.parent) if args.project else {}
        for i in range(1, 10):
            value = getattr(args, f"stage{i}")
            if value:
                paths[f"stage{i}"] = Path(value)
        try:
            if args.project and not args.stage7:
                paths['stage7'] = rebuild_project_stage7(root.parent, paths)
            if args.project and (not args.stage8 or not paths.get('stage8', Path()).is_file()):
                paths['stage8'] = rebuild_project_stage8(root.parent, paths)
            if args.project and (not args.stage9 or not paths.get('stage9', Path()).is_file()):
                paths['stage9'] = rebuild_project_stage9(root.parent, paths)
            ctx = build_pipeline_context(paths=paths, image_path=args.image, video_path=args.video)
            if args.pipeline:
                ctx.state['has_analysis_result'] = True
        except (ValueError, OSError) as exc:
            p.error(str(exc))
        server = serve_demo(ctx, args.host, args.port)
        print(f"Stage 1–9 explorer: http://{args.host}:{args.port}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
        return 0
    if args.demo:
        args.stage4 = str(root / "examples" / "stage4.json")
        args.stage7 = str(root / "examples" / "stage7.json")
        args.stage8 = str(root / "examples" / "stage8.json")
        args.stage6 = str(root / "examples" / "stage6.json")
        args.stage1 = str(root / "examples" / "stage1.json")
        args.stage3 = str(root / "examples" / "stage3.json")
        args.image = str(root / "examples" / "frame104_demo.jpg")

    missing = [name for name in ("stage4", "stage7") if not getattr(args, name)]
    if missing:
        p.error("required unless --demo: " + ", ".join("--" + x for x in missing))

    ctx = build_demo_context(
        stage4_path=args.stage4,
        stage7_path=args.stage7,
        stage8_path=args.stage8,
        stage6_path=args.stage6,
        stage1_path=args.stage1,
        stage3_path=args.stage3,
        image_path=args.image,
        video_path=args.video,
        epsilon_m=args.epsilon_m,
    )
    server = serve_demo(ctx, args.host, args.port)
    print(f"Stage 9 demo: http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
