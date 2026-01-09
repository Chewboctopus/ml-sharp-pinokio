import os
import sys
import torch
import logging
from pathlib import Path
import numpy as np
import torch.nn.functional as F

# Add ml-sharp src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
ml_sharp_src = os.path.join(current_dir, "ml-sharp", "src")
if ml_sharp_src not in sys.path:
    sys.path.append(ml_sharp_src)

from sharp.models import PredictorParams, create_predictor
from sharp.utils import io
from sharp.utils.gaussians import save_ply, SceneMetaData, unproject_gaussians
from sharp.cli.render import render_gaussians
from sharp.cli import render as sharp_render
from sharp.utils import camera
from sharp.utils import gsplat

LOGGER = logging.getLogger(__name__)

# =============================================================================
# MPS COMPATIBILITY: Patched render_gaussians
# The original ml-sharp render_gaussians only allows CUDA. We patch it to also
# support MPS for Mac ARM devices.
# =============================================================================
def render_gaussians_mps_compatible(gaussians, metadata, output_path, params=None):
    """Patched version of render_gaussians that supports both CUDA and MPS."""
    (width, height) = metadata.resolution_px
    f_px = metadata.focal_length_px

    if params is None:
        params = camera.TrajectoryParams()

    # Support both CUDA and MPS
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        raise RuntimeError("Rendering requires CUDA or MPS.")

    intrinsics = torch.tensor(
        [
            [f_px, 0, (width - 1) / 2.0, 0],
            [0, f_px, (height - 1) / 2.0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        device=device,
        dtype=torch.float32,
    )
    camera_model = camera.create_camera_model(
        gaussians, intrinsics, resolution_px=metadata.resolution_px
    )

    trajectory = camera.create_eye_trajectory(
        gaussians, params, resolution_px=metadata.resolution_px, f_px=f_px
    )
    renderer = gsplat.GSplatRenderer(color_space=metadata.color_space)
    video_writer = io.VideoWriter(output_path)

    for _, eye_position in enumerate(trajectory):
        camera_info = camera_model.compute(eye_position)
        rendering_output = renderer(
            gaussians.to(device),
            extrinsics=camera_info.extrinsics[None].to(device),
            intrinsics=camera_info.intrinsics[None].to(device),
            image_width=camera_info.width,
            image_height=camera_info.height,
        )
        color = (rendering_output.color[0].permute(1, 2, 0) * 255.0).to(dtype=torch.uint8)
        depth = rendering_output.depth[0]
        video_writer.add_frame(color, depth)
    video_writer.close()

# Patch the original module
sharp_render.render_gaussians = render_gaussians_mps_compatible
render_gaussians = render_gaussians_mps_compatible

LOGGER = logging.getLogger(__name__)

# =============================================================================
# OPTIMIZATION: Shared ViT Backbone
# Monkey-patch create_vit to reuse the same DinoV2 instance for both encoders.
# This saves ~2GB VRAM since both use the same preset.
# =============================================================================
_vit_cache = {}
_original_create_vit = None

def _install_vit_cache():
    """Install the cached create_vit wrapper."""
    global _original_create_vit
    from sharp.models.encoders import vit_encoder
    from sharp.models.encoders import monodepth_encoder
    
    if _original_create_vit is None:
        _original_create_vit = vit_encoder.create_vit
    
    def cached_create_vit(config=None, preset="dinov2l16_384", intermediate_features_ids=None):
        """Cached wrapper for create_vit. Reuses existing ViT if same preset."""
        cache_key = preset
        
        if cache_key in _vit_cache:
            model = _vit_cache[cache_key]
            # Update intermediate_features_ids if different (for patch_encoder vs image_encoder)
            # This is safe because it's just an attribute used during forward pass
            if intermediate_features_ids is not None:
                model.intermediate_features_ids = intermediate_features_ids
            LOGGER.info(f"Reusing cached ViT {preset} (intermediate_features_ids={intermediate_features_ids})")
            return model
        
        # First call - create new instance
        model = _original_create_vit(config=config, preset=preset, intermediate_features_ids=intermediate_features_ids)
        _vit_cache[cache_key] = model
        LOGGER.info(f"Created and cached ViT {preset}")
        return model
    
    # Patch both modules - vit_encoder and monodepth_encoder (which imports create_vit directly)
    vit_encoder.create_vit = cached_create_vit
    monodepth_encoder.create_vit = cached_create_vit

def _clear_vit_cache():
    """Clear the ViT cache when reloading model."""
    global _vit_cache
    _vit_cache.clear()


class MLSharpEngine:
    def __init__(self, device=None):
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
        
        self.predictor = None
        self.current_checkpoint = None
        self.low_vram = False
        LOGGER.info(f"MLSharpEngine initialized on {self.device}")

    def load_model(self, checkpoint_path=None, low_vram=False):
        """Load or reload the model with optional quantization."""
        print(f"Engine: load_model request (low_vram={low_vram})")
        if checkpoint_path is None:
            # If no path, we can't do quantization caching easily here, 
            # so we just load from hub or wait for explicit path.
            # But we still need to reload if low_vram changed!
            if self.predictor is not None and self.low_vram == low_vram:
                return
            LOGGER.info("No checkpoint path provided, model will be loaded from torch hub if possible.")
        else:
            checkpoint_path = Path(checkpoint_path)
        
        # Check if we need to reload (same checkpoint AND same low_vram mode)
        if self.predictor is not None and self.current_checkpoint == checkpoint_path and self.low_vram == low_vram:
            print("Engine: Model already loaded with same settings, skipping reload.")
            return

        final_checkpoint = checkpoint_path
        if checkpoint_path is not None:
            print(f"Engine: Loading checkpoint from {checkpoint_path} (Low VRAM: {low_vram})")
            
            # Handle quantization caching
            if low_vram and self.device == "cuda":
                quantized_path = checkpoint_path.with_name(checkpoint_path.stem + "_fp16.pt")
                if quantized_path.exists():
                    final_checkpoint = quantized_path
                    print(f"Engine: Found cached quantized model: {final_checkpoint}")
                else:
                    print("Engine: Quantizing model to FP16 and saving to cache... (This takes a moment)")
                    state_dict = torch.load(checkpoint_path, weights_only=True)
                    for k in state_dict:
                        if isinstance(state_dict[k], torch.Tensor):
                            state_dict[k] = state_dict[k].half()
                    torch.save(state_dict, quantized_path)
                    final_checkpoint = quantized_path
                    print(f"Engine: Quantized model saved to {quantized_path}")

        if final_checkpoint is not None:
            state_dict = torch.load(final_checkpoint, weights_only=True)
        else:
            # Load from hub
            from sharp.cli.predict import DEFAULT_MODEL_URL
            print(f"Engine: Downloading default model from {DEFAULT_MODEL_URL} ...")
            state_dict = torch.hub.load_state_dict_from_url(DEFAULT_MODEL_URL, progress=True)
        
        print("Engine: Building Model Architecture...")
        
        # Install ViT caching to share backbone (saves ~2GB VRAM)
        _clear_vit_cache()
        _install_vit_cache()
        
        self.predictor = create_predictor(PredictorParams())
        self.predictor.load_state_dict(state_dict)
        self.predictor.eval()
        self.predictor.to(self.device)
        print(f"Engine: Model loaded on {self.device}")
        
        # Apply FP16 if low_vram is enabled (the checkpoint might already be FP16, but .half() is idempotent)
        if low_vram and self.device == "cuda":
             self.predictor.half()
             print("Engine: Converted model to FP16 (Low VRAM mode)")
        
        self.current_checkpoint = checkpoint_path
        self.low_vram = low_vram
        LOGGER.info("Model loaded successfully")

    def get_image_focal(self, image_path):
        """Extract 35mm equivalent focal length from EXIF."""
        if not image_path: return None
        try:
            from PIL import Image
            img_pil = Image.open(image_path)
            img_exif = io.extract_exif(img_pil)
            
            f_35mm = img_exif.get("FocalLengthIn35mmFilm", img_exif.get("FocalLenIn35mmFilm", None))
            if f_35mm is None or f_35mm < 1:
                f_35mm = img_exif.get("FocalLength", None)
                if f_35mm is None:
                    return None
                if f_35mm < 10.0:
                    f_35mm *= 8.4
            return float(f_35mm)
        except Exception as e:
            LOGGER.warning(f"Error extracting focal length: {e}")
            return None

    @torch.no_grad()
    def predict(self, image_path, output_path, internal_resolution=1536, f_mm_override=None):
        """Predict Gaussians from an image and save to PLY."""
        if self.predictor is None:
            raise RuntimeError("Model not loaded. Call load_model first.")

        image_path = Path(image_path)

        if f_mm_override is not None:
             f_35mm = f_mm_override
        else:
             f_35mm = None # Trigger default logic in io.load_rgb or our own

        # Silence sharp.utils.io warning about focal length if we're overriding or just to clean up logs
        io_logger = logging.getLogger("sharp.utils.io")
        old_level = io_logger.level
        io_logger.setLevel(logging.ERROR)
        
        try:
            image, _, f_px_auto = io.load_rgb(image_path)
        finally:
            io_logger.setLevel(old_level)
        
        # If override, recalculate f_px
        if f_mm_override is not None:
             f_px = io.convert_focallength(image.shape[1], image.shape[0], f_mm_override)
        else:
             f_px = f_px_auto

        height, width = image.shape[:2]
        
        # Internal shape for the model
        internal_shape = (internal_resolution, internal_resolution)
        
        LOGGER.info(f"Running preprocessing for {image_path.name} (Internal Res: {internal_resolution})")
        image_pt = torch.from_numpy(image.copy()).float().to(self.device).permute(2, 0, 1) / 255.0
        if self.low_vram and self.device == "cuda":
            image_pt = image_pt.half()
            
        _, h, w = image_pt.shape
        disparity_factor = torch.tensor([f_px / w]).float().to(self.device)
        if self.low_vram and self.device == "cuda":
            disparity_factor = disparity_factor.half()

        image_resized_pt = F.interpolate(
            image_pt[None],
            size=(internal_shape[1], internal_shape[0]),
            mode="bilinear",
            align_corners=True,
        )

        LOGGER.info("Running inference...")
        gaussians_ndc = self.predictor(image_resized_pt, disparity_factor)

        LOGGER.info("Running postprocessing...")
        dtype = gaussians_ndc.mean_vectors.dtype
        
        intrinsics = torch.tensor(
            [
                [f_px, 0, width / 2, 0],
                [0, f_px, height / 2, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            device=self.device,
            dtype=dtype
        )
        
        intrinsics_resized = intrinsics.clone()
        intrinsics_resized[0] *= internal_shape[0] / width
        intrinsics_resized[1] *= internal_shape[1] / height

        # Ensure extrinsics also match dtype
        extrinsics = torch.eye(4, device=self.device, dtype=dtype)

        gaussians = unproject_gaussians(
            gaussians_ndc, extrinsics, intrinsics_resized, internal_shape
        )

        LOGGER.info(f"Saving 3DGS to {output_path}")
        save_ply(gaussians, f_px, (height, width), output_path)
        return gaussians, f_px, (width, height)

    def load_gaussians(self, ply_path):
        """Load Gaussians from a PLY file."""
        from sharp.utils.gaussians import load_ply
        return load_ply(Path(ply_path))

    def render_video(self, gaussians, focal_length_px, resolution_px, output_video_path, 
                     trajectory_type="rotate_forward", num_steps=60, render_depth=True):
        """Render a video from the Gaussians and return paths to generated files."""
        # Allow rendering on CUDA or MPS (Mac ARM)
        if self.device not in ("cuda", "mps"):
            LOGGER.warning(f"Rendering requires CUDA or MPS. Current device: {self.device}")
            return None, None

        lookat_mode = "point"
        if "pan_" in trajectory_type:
            lookat_mode = "ahead"
            
        params = camera.TrajectoryParams(
            type=trajectory_type,
            num_steps=num_steps,
            lookat_mode=lookat_mode
        )
        
        metadata = SceneMetaData(focal_length_px, resolution_px, "linearRGB")
        
        output_video_path = Path(output_video_path)
        # Use trajectory name in the filename to allow multiple videos
        filename_base = output_video_path.stem
        final_video_path = output_video_path.with_name(f"{filename_base}_{trajectory_type}.mp4")
        
        LOGGER.info(f"Rendering trajectory: {trajectory_type} ({num_steps} steps) to {final_video_path} (Depth: {render_depth})")
        
        # We wrap the call to ensure that if a bug exists in sharp's close(), we at least know.
        # AND we monkeypatch VideoWriter.__init__ to respect our render_depth flag
        from sharp.utils import io as sharp_io
        original_init = sharp_io.VideoWriter.__init__
        original_close = sharp_io.VideoWriter.close
        
        def patched_init(self_vw, output_path, fps=30.0, render_depth_internal=True):
            # We ignore the library's default (True) and use our passed render_depth
            return original_init(self_vw, output_path, fps=fps, render_depth=render_depth)

        def fixed_close(self_vw):
            original_close(self_vw)
            if hasattr(self_vw, 'depth_writer') and self_vw.depth_writer is not None:
                try:
                    self_vw.depth_writer.close()
                    LOGGER.info("Closed depth_writer successfully.")
                except:
                    pass
        
        # Monkeypatch only for this call
        sharp_io.VideoWriter.__init__ = patched_init
        sharp_io.VideoWriter.close = fixed_close
        
        try:
            render_gaussians(
                gaussians=gaussians,
                metadata=metadata,
                params=params,
                output_path=final_video_path
            )
        finally:
            # Restore original
            sharp_io.VideoWriter.__init__ = original_init
            sharp_io.VideoWriter.close = original_close
            
        color_video = str(final_video_path)
        depth_video = str(final_video_path.with_suffix(".depth.mp4"))
        
        if not os.path.exists(depth_video):
            depth_video = None
            
        LOGGER.info("Rendering complete")
        return color_video, depth_video

# =============================================================================
# NEW TRAJECTORIES: Dolly and Pan
# Extending ml-sharp capabilities via monkey-patching
# =============================================================================
def create_eye_trajectory_dolly_in(offset_xyz_m, distance_m, num_steps, num_repeats):
    """Dolly In: Move from 0 to +offset_z (Swapped as per user request)."""
    num_steps_total = num_steps * num_repeats
    _, _, offset_z_m = offset_xyz_m
    start_z = 0.0
    end_z = offset_z_m
    
    eye_positions = [
        torch.tensor([0.0, 0.0, z + distance_m], dtype=torch.float32)
        for z in np.linspace(start_z, end_z, num_steps_total)
    ]
    return eye_positions * num_repeats

def create_eye_trajectory_dolly_out(offset_xyz_m, distance_m, num_steps, num_repeats):
    """Dolly Out: Move from +offset_z to 0 (Swapped as per user request)."""
    num_steps_total = num_steps * num_repeats
    _, _, offset_z_m = offset_xyz_m
    start_z = offset_z_m
    end_z = 0.0
    
    eye_positions = [
        torch.tensor([0.0, 0.0, z + distance_m], dtype=torch.float32)
        for z in np.linspace(start_z, end_z, num_steps_total)
    ]
    return eye_positions * num_repeats

def create_eye_trajectory_dolly_in_out(offset_xyz_m, distance_m, num_steps, num_repeats):
    """Dolly In-Out: 0 -> -offset_z -> 0 (Sine wave)."""
    num_steps_total = num_steps * num_repeats
    _, _, offset_z_m = offset_xyz_m
    
    # We use offset_z_m for zooming. 
    # Convention: ML-Sharp Z+ is backwards. 
    # To zoom IN we go negative Z (forward), to zoom OUT we go positive Z.
    # Let's oscillate between 0 and -offset_z (zoom in) and back.
    
    eye_positions = [
        torch.tensor(
            [
                0.0,
                0.0,
                distance_m - offset_z_m * np.sin(np.pi * t), # 0 to -offset to 0
            ],
            dtype=torch.float32,
        )
        for t in np.linspace(0, num_repeats, num_steps_total)
    ]
    return eye_positions

def create_eye_trajectory_pan_left(offset_xyz_m, distance_m, num_steps, num_repeats):
    """Pan Left: Move from +offset_x to -offset_x."""
    # Camera moves +X (Right) to -X (Left), so view pans Left? 
    # Or Camera PINS Left? Usually "Pan Left" = Camera rotates/moves left.
    # Let's assume linear movement on X axis from Right to Left.
    num_steps_total = num_steps * num_repeats
    offset_x_m, _, _ = offset_xyz_m
    
    eye_positions = [
        torch.tensor([x, 0.0, distance_m], dtype=torch.float32)
        for x in np.linspace(offset_x_m, -offset_x_m, num_steps_total)
    ]
    return eye_positions * num_repeats

def create_eye_trajectory_pan_right(offset_xyz_m, distance_m, num_steps, num_repeats):
    """Pan Right: Move from -offset_x to +offset_x."""
    num_steps_total = num_steps * num_repeats
    offset_x_m, _, _ = offset_xyz_m
    
    eye_positions = [
        torch.tensor([x, 0.0, distance_m], dtype=torch.float32)
        for x in np.linspace(-offset_x_m, offset_x_m, num_steps_total)
    ]
    return eye_positions * num_repeats

def create_eye_trajectory_pan_left_right(offset_xyz_m, distance_m, num_steps, num_repeats):
    """Pan Left-Right: 0 -> -offset -> +offset -> 0 (Sine wave)."""
    num_steps_total = num_steps * num_repeats
    offset_x_m, _, _ = offset_xyz_m
    
    # Oscillate on X axis: 0 -> Left (-x) -> Right (+x) -> 0
    # sin(2*pi*t) goes 0 -> 1 -> 0 -> -1 -> 0. 
    # We want 0 -> -offset -> +offset -> 0? Or just left/right smooth?
    # Let's replicate 'shake' but only horizontal and smoother.
    
    eye_positions = [
        torch.tensor(
            [
                offset_x_m * np.sin(2 * np.pi * t), 
                0.0,
                distance_m,
            ],
            dtype=torch.float32,
        )
        for t in np.linspace(0, num_repeats, num_steps_total)
    ]
    return eye_positions

# Store original function before patching
_original_create_eye_trajectory = camera.create_eye_trajectory

def create_eye_trajectory_extended(scene, params, resolution_px, f_px):
    """Extended factory that handles new trajectory types."""
    # Check if it's one of our new types
    new_types = {
        "dolly_in": create_eye_trajectory_dolly_in,
        "dolly_out": create_eye_trajectory_dolly_out,
        "dolly_in_out": create_eye_trajectory_dolly_in_out,
        "pan_left": create_eye_trajectory_pan_left,
        "pan_right": create_eye_trajectory_pan_right,
        "pan_left_right": create_eye_trajectory_pan_left_right,
    }
    
    if params.type in new_types:
        max_offset_xyz_m = camera.compute_max_offset(scene, params, resolution_px, f_px)
        return new_types[params.type](
            max_offset_xyz_m, params.distance_m, params.num_steps, params.num_repeats
        )
    else:
        # Fallback to original for standard types
        return _original_create_eye_trajectory(scene, params, resolution_px, f_px)

# Apply monkey-patch
camera.create_eye_trajectory = create_eye_trajectory_extended
