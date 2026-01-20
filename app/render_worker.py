# render_worker.py
# Isolated subprocess worker for rendering a single trajectory
# This approach forces MPS to release memory when the process exits

import os
import sys
import json
import argparse

# Set MPS memory limit before importing torch
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ply_path', required=True)
    parser.add_argument('--output_path', required=True)
    parser.add_argument('--trajectory', required=True)
    parser.add_argument('--num_steps', type=int, default=60)
    parser.add_argument('--focal_length_px', type=float, required=True)
    parser.add_argument('--resolution_w', type=int, required=True)
    parser.add_argument('--resolution_h', type=int, required=True)
    parser.add_argument('--target_short_edge', type=int, default=None)
    args = parser.parse_args()
    
    # Import heavy modules after setting up environment
    import torch
    
    # Add ml-sharp to path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    ml_sharp_src = os.path.join(current_dir, "ml-sharp", "src")
    if ml_sharp_src not in sys.path:
        sys.path.append(ml_sharp_src)
    
    from sharp.utils.gaussians import load_ply
    from pathlib import Path
    
    # Import our engine components
    from sharp_engine import render_gaussians_mps_compatible, gsplat
    from sharp.utils import camera
    from collections import namedtuple
    
    print(f"[Worker] Loading PLY: {args.ply_path}")
    gaussians, metadata = load_ply(Path(args.ply_path))
    
    # Calculate render resolution
    render_res = (args.resolution_w, args.resolution_h)
    render_f_px = args.focal_length_px
    
    if args.target_short_edge and args.target_short_edge > 0:
        w, h = render_res
        short_edge = min(w, h)
        scale_factor = args.target_short_edge / short_edge
        
        new_w = int(w * scale_factor)
        new_h = int(h * scale_factor)
        new_w = round(new_w / 16) * 16
        new_h = round(new_h / 16) * 16
        
        real_scale = ((new_w / w) + (new_h / h)) / 2.0
        render_res = (new_w, new_h)
        render_f_px = args.focal_length_px * real_scale
        print(f"[Worker] Scaling: {args.resolution_w}x{args.resolution_h} -> {new_w}x{new_h}")
    
    # Setup trajectory params
    lookat_mode = "point"
    if "pan_" in args.trajectory:
        lookat_mode = "ahead"
    
    params = camera.TrajectoryParams(
        type=args.trajectory,
        num_steps=args.num_steps,
        lookat_mode=lookat_mode
    )
    
    SceneMetaData = namedtuple('SceneMetaData', ['focal_length_px', 'resolution_px', 'color_space'])
    scene_metadata = SceneMetaData(
        focal_length_px=render_f_px,
        resolution_px=render_res,
        color_space="linearRGB"
    )
    
    output_path = Path(args.output_path)
    final_path = output_path.with_name(f"{output_path.stem}_{args.trajectory}.mp4")
    
    print(f"[Worker] Rendering trajectory: {args.trajectory}")
    print(f"[Worker] Output: {final_path}")
    
    render_gaussians_mps_compatible(
        gaussians=gaussians,
        metadata=scene_metadata,
        params=params,
        output_path=final_path
    )
    
    # Output result as JSON for parent process
    result = {
        "status": "success",
        "color_video": str(final_path),
        "depth_video": str(final_path.with_suffix(".depth.mp4"))
    }
    print(f"[Worker] RESULT_JSON:{json.dumps(result)}")
    
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        import traceback
        traceback.print_exc()
        result = {"status": "error", "error": str(e)}
        print(f"[Worker] RESULT_JSON:{json.dumps(result)}")
        sys.exit(1)
