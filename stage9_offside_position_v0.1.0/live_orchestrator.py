import argparse
import sys
from pathlib import Path

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--frame", type=int, required=True)
    p.add_argument("--outdir", required=True)
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    print(f"[Orchestrator] Starting live analysis for frame={args.frame} image={args.image}")
    
    frame_path = Path(args.image)
    if not frame_path.exists():
        print("[Orchestrator] Failed to find uploaded frame")
        sys.exit(1)
        
    print(f"[Orchestrator] Extraction successful: {frame_path}")
    
    # Run the real pipeline. Stage 4 delegates to stage4_metric3d_v0.2.0 and
    # creates a new SAM3D native artifact for this request.
    import subprocess
    infer_script = Path(__file__).resolve().parent / 'live_single_frame_inference.py'
    print(f"[Orchestrator] Launching {infer_script} for Real Inference...")
    subprocess.check_call([sys.executable, str(infer_script), str(frame_path), str(outdir), str(args.frame)])

    print("[Orchestrator] Processing complete!")
    
if __name__ == '__main__':
    main()
