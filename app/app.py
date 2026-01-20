import logging
logging.basicConfig(level=logging.INFO)
print("--- WebUI for ML-Sharp Starting ---")
import gradio as gr
print("Gradio imported.")
import subprocess
import os
import shutil
import time
import datetime
import glob
import sys
import torch # Only for Device check
device_type = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Torch version: {torch.__version__} (Device: {device_type.upper()})")
import json
from pathlib import Path
print("Loading Engine...")
from sharp_engine import MLSharpEngine, force_cleanup
from PIL import Image

# Initialize Engine
engine = MLSharpEngine()
print("Engine Initialized.")

# Global Version
CURRENT_VERSION = "0.3"

# Global for job cancellation
import threading
current_job_stop_event = threading.Event()

def stop_generation():
    print("UI: User clicked STOP button. Sending signal...")
    current_job_stop_event.set()
    return "[Cancelled] Stopping current tasks... Please wait."

# --- PATHS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(BASE_DIR, "config.json")

OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
TEMP_DIR = os.path.join(BASE_DIR, "temp_proc")

print(f"DEBUG: BASE_DIR: {BASE_DIR}")
print(f"DEBUG: CONFIG_FILE: {CONFIG_FILE}")

# --- EXEC DETECTION ---
if sys.platform == "win32":
    SHARP_EXE = os.path.join(sys.prefix, "Scripts", "sharp.exe")
else:
    SHARP_EXE = os.path.join(sys.prefix, "bin", "sharp")

if not os.path.exists(SHARP_EXE):
    SHARP_EXE = "sharp"

# Clean init
for d in [OUTPUTS_DIR, TEMP_DIR]:
    os.makedirs(d, exist_ok=True)

def clean_gradio_cache():
    import tempfile
    # Gradio uses tempfile.gettempdir() / "gradio" by default or GRADIO_TEMP_DIR env var
    gradio_tmp = os.environ.get("GRADIO_TEMP_DIR")
    if not gradio_tmp:
        gradio_tmp = os.path.join(tempfile.gettempdir(), "gradio")
    
    print(f"DEBUG: Cleaning Gradio cache at {gradio_tmp}")
    if os.path.exists(gradio_tmp):
        try:
            # Remove the whole directory and recreate it to ensure it's empty
            shutil.rmtree(gradio_tmp)
            os.makedirs(gradio_tmp, exist_ok=True)
            print("DEBUG: Gradio cache cleared successfully.")
        except Exception as e:
            print(f"DEBUG: Error clearing Gradio cache: {e}")

def clean_temp_dir():
    print(f"DEBUG: Cleaning temp_proc at {TEMP_DIR}")
    if os.path.exists(TEMP_DIR):
        try:
            # Remove files inside but keep the directory
            for f in glob.glob(os.path.join(TEMP_DIR, "*")):
                if os.path.isfile(f) or os.path.islink(f):
                    os.unlink(f)
                elif os.path.isdir(f):
                    shutil.rmtree(f)
            print("DEBUG: temp_proc cleared successfully.")
        except Exception as e:
            print(f"DEBUG: Error clearing temp_proc: {e}")

# Run cleanup on startup
clean_gradio_cache()
clean_temp_dir()

# --- PLY CONVERSION FOR GRADIO ---
import numpy as np

def convert_ply_for_gradio(input_path: str) -> str:
    """
    Optimized PLY conversion:
    1. Saves to TEMP_DIR to keep job folder clean.
    2. Caches result based on timestamp.
    3. Renames file to include JOB NAME for easier identification upon download.
    """
    try:
        from plyfile import PlyData, PlyElement
    except ImportError:
        print("DEBUG: plyfile not installed, skipping PLY conversion")
        return input_path
    
    if not os.path.exists(input_path):
        return input_path
        
    # --- SMART NAMING LOGIC ---
    # Extract Job Name from parent folder (e.g. "outputs/my_job_123/input.ply" -> "my_job_123")
    job_name = os.path.basename(os.path.dirname(input_path))
    file_name = os.path.basename(input_path)
    base_name, ext = os.path.splitext(file_name)
    mtime = int(os.path.getmtime(input_path))
    
    # New Format: JobName_OriginalName_view_Timestamp.ply
    # Example: photo-1646_input_source_view_176873.ply
    temp_output_name = f"{job_name}_{base_name}_view_{mtime}{ext}"
    output_path = os.path.join(TEMP_DIR, temp_output_name)
    
    # CACHE CHECK
    if os.path.exists(output_path):
        # print(f"DEBUG: Using cached PLY: {output_path}")
        return output_path
    
    try:
        # print(f"DEBUG: Converting PLY for Gradio: {input_path}")
        plydata = PlyData.read(input_path)
        vertex = plydata['vertex']
        
        props = vertex.data.dtype.names
        num_points = len(vertex.data)
        
        # Strip to float32
        new_dtype = [(name, 'f4') for name in props]
        new_data = np.empty(num_points, dtype=new_dtype)
        
        for name in props:
            new_data[name] = vertex.data[name].astype(np.float32)
        
        new_element = PlyElement.describe(new_data, 'vertex')
        PlyData([new_element], text=False).write(output_path)
        
        print(f"DEBUG: PLY converted and cached as: {temp_output_name}")
        return output_path
        
    except Exception as e:
        print(f"DEBUG: PLY conversion failed: {e}")
        return input_path

# --- HELPERS ---

def run_command_generator(command, cwd=None, env=None):
    """
    Generator that runs a shell command and yields output lines in real-time.
    """
    print(f"DEBUG: Executing: {command}")
    
    # Merge env with system env
    full_env = os.environ.copy()
    if env:
        full_env.update(env)
        
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, # Merge stderr into stdout
        shell=True,
        text=True,
        cwd=cwd,
        env=full_env,
        bufsize=1, # Line buffered
        universal_newlines=True,
        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    )
    
    # Yield output line by line
    for line in process.stdout:
        yield line.strip()
        
    process.wait()
    if process.returncode != 0:
        yield f"DEBUG: Process failed with code {process.returncode}"
        raise subprocess.CalledProcessError(process.returncode, command)


def check_cuda():
    """Legacy name kept for compatibility - actually checks gsplat availability."""
    return check_gsplat()

def check_gsplat():
    """Check if gsplat is installed (enables video rendering on CUDA or MPS)."""
    try:
        import gsplat
        # print(f"DEBUG: gsplat found, version: {getattr(gsplat, '__version__', 'unknown')}")
        return True
    except ImportError:
        print("DEBUG: gsplat NOT installed - video rendering disabled")
        return False

def load_config():
    print(f"DEBUG: Loading config from {CONFIG_FILE}")
    if not os.path.exists(CONFIG_FILE): 
        print("DEBUG: Config file not found, returning empty.")
        return {}
    try:
        with open(CONFIG_FILE, "r") as f:
            data = json.load(f)
            print(f"DEBUG: Loaded config data: {data}")
            return data
    except Exception as e:
        print(f"DEBUG: Error loading config: {e}")
        return {}

def save_config(key, value):
    # Ensure at least one trajectory if we are saving that key
    if key == "trajectory_type" and (not value or len(value) == 0):
        print("DEBUG: Ignoring attempt to save empty trajectory list.")
        return

    print(f"DEBUG: Saving config {key}={value}")
    
    # Simple retry mechanism for concurrent access
    for i in range(5):
        try:
            cfg = load_config()
            cfg[key] = value
            # Write to a temporary file first for atomicity
            temp_cfg = CONFIG_FILE + ".tmp"
            with open(temp_cfg, "w") as f:
                json.dump(cfg, f, indent=4)
            os.replace(temp_cfg, CONFIG_FILE)
            print("DEBUG: Config saved successfully.")
            break
        except Exception as e:
            print(f"DEBUG: Save config attempt {i+1} failed: {e}")
            time.sleep(0.1)

def save_metadata(job_dir, data):
    meta_path = os.path.join(job_dir, "job_info.json")
    try:
        with open(meta_path, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"Error saving metadata: {e}")

def load_metadata(job_dir):
    meta_path = os.path.join(job_dir, "job_info.json")
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r") as f:
                return json.load(f)
        except:
            pass
    return {}



def get_history_list():
    """
    Return a list of tuples (thumb_path, caption) for the Gallery.
    """
    items = []
    if not os.path.exists(OUTPUTS_DIR): 
        return []
    
    subdirs = [f.path for f in os.scandir(OUTPUTS_DIR) if f.is_dir()]
    subdirs.sort(key=os.path.getmtime, reverse=True)
    
    for job_dir in subdirs:
        job_name = os.path.basename(job_dir)
        meta = load_metadata(job_dir)
        
        date_str = meta.get("date", "")
        if not date_str:
            ts = os.path.getmtime(job_dir)
            date_str = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M')
            
        orig_name = meta.get("original_name", "")
        
        caption = f"{job_name}\nDate: {date_str}\nInput: {orig_name}"
        
        # Find thumbnail
        thumb = None
        candidates = sorted(glob.glob(os.path.join(job_dir, "*input*.*")))
        if not candidates:
            candidates = sorted(glob.glob(os.path.join(job_dir, "*.*")))
            
        for f in candidates:
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                thumb = f
                break
        
        if thumb:
            items.append((thumb, caption))
            
    return items

def generate_job_list_html():
    """Generate HTML for job list with clickable rows."""
    items = get_history_list()
    
    if not items:
        return '<div class="job-list-container"><p style="text-align:center;color:gray;">No jobs yet.</p></div>'
    
    html_parts = ['<div class="job-list-container">']
    
    for thumb_path, caption in items:
        lines = caption.split('\n')
        job_name = lines[0] if lines else "Unknown"
        date_str = lines[1].replace("Date: ", "") if len(lines) > 1 else ""
        input_str = lines[2].replace("Input: ", "") if len(lines) > 2 else ""
        
        # Use Gradio's /gradio_api/file= endpoint for local files (like Wan2GP)
        thumb_url = "/gradio_api/file=" + thumb_path.replace("\\", "/")
        
        # Escape single quotes in job name for JavaScript
        safe_job_name = job_name.replace("'", "\\'")
        
        html_parts.append(f'''
        <div class="job-list-item" data-job="{safe_job_name}" onclick="selectJob('{safe_job_name}')">
            <img src="{thumb_url}" alt="thumb" onerror="this.alt='[No Image]'"/>
            <div class="job-info">
                <div class="job-name" title="{job_name}">{job_name}</div>
                <div class="job-meta">{date_str}</div>
                <div class="job-meta" title="{input_str}">Input: {input_str}</div>
            </div>
            <button class="delete-btn" onclick="event.stopPropagation(); deleteJob('{safe_job_name}')" title="Delete Job">✕</button>
        </div>
        ''')
    
    html_parts.append('</div>')
    
    # Note: JavaScript selectJob and deleteJob functions are defined globally in head_js
    
    return ''.join(html_parts)


def get_input_library_items():
    """Dummy function to prevent NameError if still referenced."""
    return []

# --- RESOLUTION & IMAGE HELPERS ---

def get_img_resolution_data(image_path):
    """
    Returns (width, height, megapixel, display_text).
    """
    if not image_path:
        return 0, 0, 0, "Input Resolution: N/A"
    try:
        with Image.open(image_path) as img:
            w, h = img.size
            mp = (w * h) / 1_000_000
            return w, h, mp, f"Input Resolution: {w}x{h} ({mp:.1f} MP)"
    except Exception:
        return 0, 0, 0, "Input Resolution: Error"

def get_smart_resolution_choices(img_w, img_h):
    """
    Generates a list of resolution options including 960p and 640p.
    Sorts 'Original' by actual size and adds High Resources warnings.
    """
    short_edge = min(img_w, img_h) if (img_w > 0 and img_h > 0) else 0
    
    # Define standard presets (Label, ShortEdgeSize)
    presets = [
        ("4K (2160p)", 2160),
        ("QHD (1440p)", 1440),
        ("FHD (1080p)", 1080),
        ("960p", 960),
        ("HD (720p)", 720),
        ("640p", 640),
        ("SD (480p)", 480)
    ]
    
    # Create the "Original" entry with its real size for sorting
    if short_edge > 0:
        orig_label = f"Original ({short_edge}p)"
        presets.append((orig_label, short_edge))
    else:
        presets.append(("Original", 9999)) 

    # Sort by size descending
    presets.sort(key=lambda x: x[1], reverse=True)
    
    final_choices = []
    default_value = None
    
    for label, size in presets:
        display_str = label
        if size > 1088:
            display_str += " ⚠️ (High Resources)"
        final_choices.append(display_str)
        
        # Smart default selection logic: Prefer FHD (1080p)
        if "1080p" in label and default_value is None:
            default_value = display_str
    
    if default_value is None and final_choices:
        default_value = final_choices[0]
            
    return final_choices, default_value

def parse_resolution_string(res_str):
    """Extract integer resolution from string format like 'FHD (1080p)' or '960p'."""
    if not res_str or "Original" in res_str:
        return None
    import re
    # Match any digits followed by 'p' (handles parens or no parens)
    match = re.search(r"(\d+)p", res_str)
    if match:
        return int(match.group(1))
    return None

def get_job_input_image(job_dir):
    """Finds the input image in the job folder (searches for 'input' in the name or takes the first image)."""
    if not os.path.exists(job_dir): return None
    
    candidates = glob.glob(os.path.join(job_dir, "*"))
    # Priority 1: File with "input" in the name
    for f in candidates:
        if "input" in os.path.basename(f).lower() and f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
            return f
    # Priority 2: Any image
    for f in candidates:
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
            return f
    return None

def get_target_dimensions(w, h, res_str):
    """
    CORE LOGIC: Calculates the final dimensions (W, H) based on the resolution string.
    Centralizes the rules of 'Smart Tolerance' and 'Modulo 16' in a single point.
    """
    if w == 0 or h == 0: return 0, 0
    
    target_short = parse_resolution_string(res_str)
    
    new_w, new_h = w, h
    
    # Logic of Scale
    if target_short is not None:
        short_edge = min(w, h)
        if short_edge == 0: return 0, 0
        
        scale_factor = target_short / short_edge
        
        # Rule 1: Smart Tolerance (< 2% diff -> keeps original)
        if abs(1.0 - scale_factor) < 0.02:
            scale_factor = 1.0
            
        new_w = int(w * scale_factor)
        new_h = int(h * scale_factor)
            
    # Rule 2: Force Modulo 16 (Rounds to the nearest multiple of 16)
    new_w = round(new_w / 16) * 16
    new_h = round(new_h / 16) * 16
    
    return new_w, new_h

def calculate_safe_target_resolution(image_path, res_str):
    """
    Helper Backend: Uses the core logic to tell the Engine which short side to use.
    """
    if not image_path or not os.path.exists(image_path):
        return None # Let the engine decide

    try:
        with Image.open(image_path) as img:
            w, h = img.size
    except:
        return None

    nw, nh = get_target_dimensions(w, h, res_str)
    return min(nw, nh)

def calc_final_resolution_text(img_w, img_h, res_str):
    """
    Helper Frontend: Uses the core logic to show the UI label.
    """
    if img_w == 0 or img_h == 0:
        return "Output Resolution: N/A"
        
    nw, nh = get_target_dimensions(img_w, img_h, res_str)
    
    mp = (nw * nh) / 1_000_000
    return f"Output Resolution: {nw}x{nh} ({mp:.1f} MP)"

# --- THREADED EXECUTION WRAPPER ---
import threading
import queue

class QueuedLoggingHandler(logging.Handler):
    def __init__(self, log_queue):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record):
        try:
            msg = self.format(record)
            self.log_queue.put(msg)
        except Exception:
            self.handleError(record)

def append_log_smart(current_log, new_msg):
    """
    Appends new_msg to current_log with smart replacement for progress lines.
    If new_msg is a percentage update (contains '(%') for the same task,
    it replaces the last line instead of appending.
    """
    if not current_log:
        return new_msg + "\n"
    
    lines = current_log.strip().split("\n")
    if not lines:
        return new_msg + "\n"
        
    last_line = lines[-1]
    
    # Check if both last_line and new_msg are progress updates
    is_prev_progress = "(%" in last_line
    is_new_progress = "(%" in new_msg
    
    if is_prev_progress and is_new_progress:
        # Check if they belong to the same task (e.g. [rotate_forward])
        # Progress msg format from engine: "[traj_type] [Rendering] X/Y frames (Z%)"
        prev_task = last_line.split("]")[0] if "]" in last_line else ""
        new_task = new_msg.split("]")[0] if "]" in new_msg else ""
        
        if prev_task == new_task:
            # Replace last line
            lines[-1] = new_msg
            return "\n".join(lines) + "\n"
    
    # Standard append
    return current_log + new_msg + "\n"
        

def predict(image_path, do_render_video, resolution_setting, traj_config=None, num_steps=60, low_vram=False, f_mm=None, progress=gr.Progress()):
    # Reset stop event at start of job
    current_job_stop_event.clear()
    if not image_path: 
        yield ("error", "Error: No image uploaded.")
        return

    # Queue for logs - with max size to prevent memory buildup
    log_queue = queue.Queue(maxsize=100)
    handler = QueuedLoggingHandler(log_queue)
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)
    
    # Use dict to avoid generator closure retention
    result_container = {}
    
    def target():
        try:
            ts = int(time.time())
            original_name = os.path.basename(image_path)
            safe_name, ext = os.path.splitext(original_name)
            
            job_name = f"{safe_name}_{ts}"
            job_dir = os.path.join(OUTPUTS_DIR, job_name)
            os.makedirs(job_dir, exist_ok=True)
            
            job_input = os.path.join(job_dir, f"input_source{ext}")
            shutil.copy(image_path, job_input)
            
            print(f"START JOB (Engine): {job_name}")
            log_queue.put("[Training] Starting analysis...")
            
            meta = {
                "job_name": job_name,
                "original_name": original_name,
                "timestamp": ts,
                "date": datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S'),
                "status": "processing"
            }
            save_metadata(job_dir, meta)
            
            lv = low_vram
            if lv is None:
                config = load_config()
                lv = config.get("low_vram", False)
            
            resolution = 1536
            
            checkpoint_path = os.path.join(BASE_DIR, "app", "ml-sharp", "sharp_2572gikvuh.pt")
            if not os.path.exists(checkpoint_path):
                 checkpoint_path = os.path.join(BASE_DIR, "sharp_2572gikvuh.pt")
            
            if not os.path.exists(checkpoint_path):
                checkpoint_path = None
            else:
                checkpoint_path = Path(checkpoint_path)

            engine.load_model(checkpoint_path, low_vram=lv)
            
            # Predict PLY
            if current_job_stop_event.is_set():
                print("Engine: Stop signal detected BEFORE PLY generation.")
                log_queue.put("[Cancelled] Job stopped by user")
                result_container['success'] = True
                result_container['data'] = (f"[Cancelled] Job stopped early: {job_name}", None, None, None, job_dir)
                return
            
            ply_output_path = os.path.join(job_dir, f"input_source.ply")
            gaussians, f_px, res_px = engine.predict(job_input, ply_output_path, internal_resolution=resolution, f_mm_override=f_mm)
            
            # NOTIFY PLY READY
            log_queue.put(("ply_ready", (ply_output_path, f_px, res_px)))
            
            # Critical: If no video rendering, cleanup gaussians immediately
            if not (do_render_video and check_cuda()):
                del gaussians
                force_cleanup()
            
            # Don't store video paths in generator scope
            
            if do_render_video and check_cuda():
                local_traj_config = traj_config
                if not local_traj_config:
                     local_traj_config = {"rotate_forward": {"enabled": True, "depth": True}}
                
                enabled_trajs = [(k, v["depth"]) for k, v in local_traj_config.items() if v.get("enabled", False)]
                if not enabled_trajs:
                     enabled_trajs = [("rotate_forward", True)]
                
                vid_color_path = os.path.join(job_dir, "input_source.mp4")
                rendered_videos = [] 
                
                # Parse Resolution
                target_short = calculate_safe_target_resolution(image_path, resolution_setting)
                
                # Use subprocess isolation for all GPU rendering (better memory management)
                # Each trajectory runs in isolated process that releases ALL GPU memory on exit
                use_subprocess = engine.device in ("mps", "cuda")
                
                print(f"Engine: Rendering {len(enabled_trajs)} trajectories... (subprocess={use_subprocess})")
                for traj_type, do_depth in enabled_trajs:
                    log_queue.put(("rendering", (traj_type, do_depth)))
                    print(f"Engine: Rendering {traj_type} - Depth: {do_depth}...")
                    
                    # Final check before each subprocess/render starts
                    if current_job_stop_event.is_set():
                        print("Engine: Stop signal detected. Cancelling remaining trajectories.")
                        break

                    # Create progress callback for Gradio log updates
                    def progress_cb(msg):
                        log_queue.put(("progress", msg))
                    
                    # Pre-calculate expected paths for robust cleanup
                    expected_vc = os.path.join(job_dir, f"input_source_{traj_type}.mp4")
                    expected_vd = os.path.join(job_dir, f"input_source_{traj_type}.depth.mp4")
                    if traj_type == "rotate_forward" and not os.path.exists(expected_vc):
                        # Fallback for default name
                        expected_vc = os.path.join(job_dir, "input_source.mp4")
                        expected_vd = os.path.join(job_dir, "input_source.depth.mp4")

                    if use_subprocess:
                        # Use subprocess isolation for MPS to release memory after each trajectory
                        vc, vd = engine.render_video_subprocess(
                            ply_output_path, f_px, res_px, vid_color_path,
                            trajectory_type=traj_type, num_steps=num_steps,
                            render_depth=do_depth, target_short_edge=target_short,
                            progress_callback=progress_cb,
                            stop_event=current_job_stop_event
                        )
                    else:
                        # Standard in-process render for CUDA
                        vc, vd = engine.render_video(
                            gaussians, f_px, res_px, vid_color_path, 
                            trajectory_type=traj_type, num_steps=num_steps, 
                            render_depth=do_depth, target_short_edge=target_short,
                            stop_event=current_job_stop_event
                        )
                    
                    if current_job_stop_event.is_set():
                        # Cleanup partial files using pre-calculated paths
                        for p in [expected_vc, expected_vd]:
                            if p and os.path.exists(p):
                                try: os.remove(p)
                                except: pass
                        print(f"Engine: Cleaned up partial files for {traj_type}")
                        break

                    rendered_videos.append(traj_type)
                    
                    # Yield immediately and don't retain paths
                    log_queue.put(("video_ready", (traj_type, vc, vd)))
                    
                    # Critical: delete path references immediately
                    del vc, vd
                    
                    # Aggressive cleanup after each video
                    force_cleanup()
                
                # CRITICAL: Delete gaussians after all renders complete
                del gaussians
                force_cleanup()

            if current_job_stop_event.is_set():
                 log_queue.put("[Cancelled] Job stopped by user")
                 result_container['success'] = True
                 result_container['data'] = (f"[Cancelled] Job stopped: {job_name}", None, None, None, job_dir)
            else:
                 log_queue.put("[Done] Job finished successfully")
                 meta["status"] = "completed"
                 if do_render_video and check_cuda():
                     meta["rendered_trajectories"] = rendered_videos
                 save_metadata(job_dir, meta)
                 
                 log_queue.put("[Done] Video generation complete")
                 result_container['success'] = True
                 result_container['data'] = (f"[Done] Completed: {job_name}", None, None, None, job_dir)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            result_container['success'] = False
            result_container['error'] = str(e)
            
    t = threading.Thread(target=target)
    t.start()
    
    current_log = ""
    while t.is_alive():
        try:
            # Non-blocking get
            raw_msg = log_queue.get_nowait()
            
            # Check if it's a special event
            if isinstance(raw_msg, tuple):
                event_type, event_data = raw_msg
                
                # Handle progress updates by adding to log
                if event_type == "progress":
                    current_log = append_log_smart(current_log, event_data)
                    # We yield 'log' so the UI updates the textbox
                    # The JS will see [Rendering] or [Done] tags inside the log text
                    yield ("log", current_log)
                else:
                    yield (event_type, event_data)
            else:
                msg = raw_msg
                current_log = append_log_smart(current_log, msg)
                yield ("log", current_log)
                
        except queue.Empty:
            time.sleep(0.1)
            
    while not log_queue.empty():
        raw_msg = log_queue.get()
        if isinstance(raw_msg, tuple):
             event_type, event_data = raw_msg
             yield (event_type, event_data)
        else:
             msg = raw_msg
             current_log = append_log_smart(current_log, msg)
             
    root_logger.removeHandler(handler)
    
    if result_container.get("success"):
        yield ("done", (current_log, result_container['data']))
    else:
        err = result_container.get("error", "Unknown Error")
        yield ("error", current_log + "\nError: " + err)

def render_video_gen(job_name, resolution_setting, traj_config=None, num_steps=60, low_vram=False):
    if not job_name: 
        yield ("error", "No job selected")
        return

    log_queue = queue.Queue()
    handler = QueuedLoggingHandler(log_queue)
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)
    
    result_container = {}
    
    def target():
        try:
            job_dir = os.path.join(OUTPUTS_DIR, job_name)
            plys = glob.glob(os.path.join(job_dir, "*.ply"))
            if not plys: 
                result_container['success'] = False
                result_container['error'] = "No PLY file found in this job."
                return
            
            ply_path = plys[0]
            
            if not check_gsplat():
                result_container['success'] = False
                result_container['error'] = "Video rendering requires gsplat (CUDA/MPS)."
                return

            checkpoint_path = os.path.join(BASE_DIR, "app", "ml-sharp", "sharp_2572gikvuh.pt")
            if not os.path.exists(checkpoint_path):
                 checkpoint_path = os.path.join(BASE_DIR, "sharp_2572gikvuh.pt")
            
            if not os.path.exists(checkpoint_path):
                checkpoint_path = None
            else:
                checkpoint_path = Path(checkpoint_path)
                
            engine.load_model(checkpoint_path, low_vram=low_vram)
            
            gaussians, metadata = engine.load_gaussians(ply_path)
            
            local_traj_config = traj_config
            if not local_traj_config:
                 local_traj_config = {"rotate_forward": {"enabled": True, "depth": True}}
            
            enabled_trajs = [(k, v["depth"]) for k, v in local_traj_config.items() if v.get("enabled", False)]
            if not enabled_trajs:
                 enabled_trajs = [("rotate_forward", True)]
                
            vid_color_path = os.path.join(job_dir, f"{Path(ply_path).stem}.mp4")
            
            last_vc, last_vd = None, None
            rendered_videos = []

            # --- USE HELPER & CALC RESOLUTION ---
            input_image_path = get_job_input_image(job_dir)
            target_short = calculate_safe_target_resolution(input_image_path, resolution_setting)
            
            # Use subprocess isolation for all GPU rendering (better memory management)
            # Each trajectory runs in isolated process that releases ALL GPU memory on exit
            use_subprocess = engine.device in ("mps", "cuda")
            
            log_queue.put(f"[Rendering] Starting video generation for {len(enabled_trajs)} trajectories...")
            print(f"Engine (Regen): Rendering {len(enabled_trajs)} trajectories... (subprocess={use_subprocess})")
            for traj_type, do_depth in enabled_trajs:
                # Check cancellation first
                if current_job_stop_event.is_set():
                    print("Engine (Regen): Stop signal detected. Cancelling.")
                    break

                print(f"Engine (Regen): Rendering {traj_type} ({num_steps} steps) - Depth: {do_depth}...")
                
                # Create progress callback for Gradio log updates
                def progress_cb(msg):
                    log_queue.put(("progress", msg))
                
                # Pre-calculate expected paths for robust cleanup
                expected_vc = os.path.join(job_dir, f"{Path(ply_path).stem}_{traj_type}.mp4")
                expected_vd = os.path.join(job_dir, f"{Path(ply_path).stem}_{traj_type}.depth.mp4")

                if use_subprocess:
                    # Use subprocess isolation for MPS to release memory after each trajectory
                    vc, vd = engine.render_video_subprocess(
                        ply_path, metadata.focal_length_px, metadata.resolution_px,
                        vid_color_path, trajectory_type=traj_type, num_steps=num_steps,
                        render_depth=do_depth, target_short_edge=target_short,
                        progress_callback=progress_cb,
                        stop_event=current_job_stop_event
                    )
                else:
                    # Standard in-process render for CUDA
                    vc, vd = engine.render_video(
                        gaussians, metadata.focal_length_px, metadata.resolution_px, 
                        vid_color_path, trajectory_type=traj_type, num_steps=num_steps, 
                        render_depth=do_depth, target_short_edge=target_short,
                        stop_event=current_job_stop_event
                    )
                
                if current_job_stop_event.is_set():
                    # Cleanup partial files using pre-calculated paths
                    for p in [expected_vc, expected_vd]:
                        if p and os.path.exists(p):
                            try: os.remove(p)
                            except: pass
                    print(f"Engine (Regen): Cleaned up partial files for {traj_type}")
                    break
                
                rendered_videos.append(traj_type)
                
                # Notify immediately
                log_queue.put(("video_ready", (traj_type, vc, vd)))
                
                # Critical: delete paths immediately after notification
                del vc, vd
                
                # Aggressive cleanup after each video
                force_cleanup()

            # CRITICAL: Delete gaussians after all renders complete
            del gaussians
            if metadata is not None:
                del metadata
            force_cleanup()

            meta = load_metadata(job_dir)
            meta["rendered_trajectories"] = rendered_videos
            save_metadata(job_dir, meta)

            if current_job_stop_event.is_set():
                 log_queue.put("[Cancelled] Job stopped by user")
                 result_container['success'] = True # Handled as success but with cancelled msg
                 result_container['data'] = ("[Cancelled] Stopped by user", None, None)
            else:
                 log_queue.put("[Done] Video generation complete")
                 result_container['success'] = True
                 result_container['data'] = ("Video(s) Rendered!", last_vc, last_vd)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            result_container['success'] = False
            result_container['error'] = str(e)

    t = threading.Thread(target=target)
    t.start()
    
    current_log = ""
    while t.is_alive():
        try:
            raw_msg = log_queue.get_nowait()
            
            # Check if it's a special event (tuple) or plain log string
            if isinstance(raw_msg, tuple):
                event_type, event_data = raw_msg
                if event_type == "progress":
                    current_log = append_log_smart(current_log, event_data)
                    yield ("log", current_log)
                else:
                    yield (event_type, event_data)
            else:
                current_log = append_log_smart(current_log, raw_msg)
                yield ("log", current_log)
        except queue.Empty:
            time.sleep(0.1)
            
    while not log_queue.empty():
        raw_msg = log_queue.get()
        if isinstance(raw_msg, tuple):
            event_type, event_data = raw_msg
            if event_type == "progress":
                current_log = append_log_smart(current_log, event_data)
            else:
                yield (event_type, event_data)
        else:
            current_log = append_log_smart(current_log, raw_msg)
        
    root_logger.removeHandler(handler)
    
    if result_container.get("success"):
        yield ("done", (current_log, result_container['data']))
    else:
        err = result_container.get("error", "Unknown Error")
        yield ("error", current_log + "\nError: " + err)

def load_job_details(evt: gr.SelectData):
    # Retrieve job name from current list
    if isinstance(evt.index, (list, tuple)):
        row_idx = evt.index[0]
    else:
        row_idx = evt.index
        
    row_idx = evt.index
        
    all_rows = get_history_list()
    if row_idx >= len(all_rows): return None, None, [], None, None, None
    
    # item is (thumb, caption)
    # caption is "JobName\n..."
    caption = all_rows[row_idx][1]
    selected_job_name = caption.split('\n')[0] 
    job_dir = os.path.join(OUTPUTS_DIR, selected_job_name)
    
    # 1. Input Img
    input_img = None
    
    # 2. Files
    files = glob.glob(os.path.join(job_dir, "*.*"))
    
    candidates = sorted(glob.glob(os.path.join(job_dir, "*input*.*")))
    if not candidates: candidates = sorted(glob.glob(os.path.join(job_dir, "*.*")))
    
    for f in candidates:
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
            input_img = f
            break
            
    vid_color = None
    vid_depth = None
    file_list = []
    ply_file = None
    
    for f in files:
        file_list.append(f)
        if f.endswith(".ply"): ply_file = f
        if f.endswith("input_source.mp4"): vid_color = f
        if f.endswith("depth.mp4"): vid_depth = f
        
    # Status Video Button
    vid_status_msg = ""
    if vid_color or vid_depth:
        vid_status_msg = "Video available."
        vid_btn_interactive = True # Può voler rigenerare
    else:
        vid_status_msg = "No video present."
        vid_btn_interactive = True
        

    return (
        selected_job_name,        # Label
        input_img,                # Image Preview
        file_list,                # File links
        vid_color,                # Video Color
        vid_depth,                # Video Depth
        f"Status: {vid_status_msg}" # Log
    )

# --- JOB HELPERS (Global Scope) ---

# Global state for task tracking
running_tasks = {}

def is_system_busy():
    return len(running_tasks) > 0

def get_busy_message():
    if not running_tasks: return ""
    # Return first active task status
    job, status = next(iter(running_tasks.items()))
    return f"System Busy: Processing '{job}' ({status})"

# --- JOB HELPERS (Global Scope) ---
def zip_job(job_name):
    # helper for error return (refresh but keep zip hidden)
    def return_error():
         return load_job_details_by_name(job_name)

    if not job_name: 
        yield return_error()
        return
        
    if is_system_busy():
        print(f"DEBUG: System busy, rejecting zip_job for {job_name}")
        yield return_error()
        return

    job_dir = os.path.join(OUTPUTS_DIR, job_name)
    if not os.path.exists(job_dir): 
        yield return_error()
        return
    
    # Set running state
    running_tasks[job_name] = "Creating ZIP archive..."
    
    try:
        # Yield 1: Processing
        yield (
            gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(),
            running_tasks[job_name], 
            gr.update(visible=False), 
            gr.update(interactive=False), gr.update(interactive=False), # Lock other actions
            gr.update(interactive=False), # Lock self (zip)
            gr.update(interactive=False), # Lock delete
            gr.update(interactive=False)  # Lock global run
        )
        
        # Create zip in TEMP_DIR
        zip_base = os.path.join(TEMP_DIR, job_name)
        archive_path = shutil.make_archive(zip_base, 'zip', job_dir)
        full_path = archive_path
        
        # Clear state
        if job_name in running_tasks: del running_tasks[job_name]
        
        # Get standard refresh state
        fresh_state = list(load_job_details_by_name(job_name))
        
        # Override zip output (index 8)
        fresh_state[8] = gr.update(value=full_path, visible=True, label=f"Download {os.path.basename(full_path)}")
        fresh_state[7] = "ZIP Created Successfully"
        
        yield tuple(fresh_state)
        
    except Exception as e:
        print(f"DEBUG: Error creating zip: {e}")
        if job_name in running_tasks: del running_tasks[job_name]
        yield return_error()

def retry_job_ply(job_name, low_vram, progress=gr.Progress()):
    def return_refresh(log_msg=""):
        res = list(load_job_details_by_name(job_name))
        if log_msg: res[7] = log_msg
        return tuple(res)
        
    if not job_name: 
        yield return_refresh("No job selected")
        return
        
    if is_system_busy():
        yield return_refresh(get_busy_message())
        return

    job_dir = os.path.join(OUTPUTS_DIR, job_name)
    input_file = get_job_input_image(job_dir) # Clean Helper
    
    if not input_file:
        yield return_refresh("Error: No valid input image found.")
        return
        
    running_tasks[job_name] = "Generating PLY..."
    
    # Threaded Execution for PLY
    log_queue = queue.Queue()
    handler = QueuedLoggingHandler(log_queue)
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)
    
    result_container = {}

    def target():
        try:
             # Load Model
            checkpoint_path = os.path.join(BASE_DIR, "app", "ml-sharp", "sharp_2572gikvuh.pt")
            if not os.path.exists(checkpoint_path):
                 checkpoint_path = os.path.join(BASE_DIR, "sharp_2572gikvuh.pt")
            
            if not os.path.exists(checkpoint_path):
                checkpoint_path = None
            else:
                checkpoint_path = Path(checkpoint_path)

            engine.load_model(checkpoint_path, low_vram=low_vram)
            
            # Predict
            ply_output_path = os.path.join(job_dir, "input_source.ply")
            engine.predict(input_file, ply_output_path, internal_resolution=1536)
            
            result_container['success'] = True
            
        except Exception as e:
            traceback.print_exc()
            result_container['success'] = False
            result_container['error'] = str(e)

    t = threading.Thread(target=target)
    t.start()
    
    # Initial Yield with marker for JS animation
    yield (
        gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), # 0-6
        "[Generating] " + running_tasks[job_name] + " (Queued)...",  # 7 (Log)
        gr.skip(), # 8 (Zip)
        gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False), # 9-13 (Buttons)
        gr.skip(), gr.skip(), gr.skip(), # 14-16 (New Res Components)
        gr.update(interactive=True), gr.update(interactive=True) # 17-18 (Stop Buttons)
    )

    current_log = ""
    while t.is_alive():
        try:
            msg = log_queue.get_nowait()
            current_log += msg + "\n"
            # Yield Log Update
            res = list(load_job_details_by_name(job_name))
            res[7] = current_log
            if progress: progress(None, desc="Generating PLY...")
            yield tuple(res)
        except queue.Empty:
            time.sleep(0.1)
            
    while not log_queue.empty():
        msg = log_queue.get()
        current_log += msg + "\n"
        
    root_logger.removeHandler(handler)
    
    if job_name in running_tasks: del running_tasks[job_name]
    
    if result_container.get("success"):
        yield return_refresh(f"{current_log}\n[Done] PLY Generated Successfully!")
    else:
        yield return_refresh(f"{current_log}\n[Done] Error: {result_container.get('error')}")

def regen_video_action(job_name, resolution_setting, low_vram, seconds, *traj_args, progress=gr.Progress()):
    def return_refresh(log_msg=""):
        res = list(load_job_details_by_name(job_name))
        if log_msg: res[7] = log_msg
        return tuple(res)
        
    if not job_name: 
        yield return_refresh("No job selected")
        return
        
    if is_system_busy():
        yield return_refresh(get_busy_message())
        return
        
    keys = ["rotate_forward", "rotate", "swipe", "shake", "dolly_in", "dolly_out", "dolly_in_out", "pan_left", "pan_right", "pan_left_right"]
    traj_config = {}
    for i, key in enumerate(keys):
        traj_config[key] = {"enabled": traj_args[i*2], "depth": traj_args[i*2+1]}
    
    num_steps = seconds * 30
    
    running_tasks[job_name] = "Generating Video..."
    
    try:
         current_job_stop_event.clear()
         # Initial: Loading with marker for JS animation
         yield (
             gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), # 0-6
             "[Rendering] Starting Video Render...", # 7 (Log)
             gr.skip(), # 8 (Zip)
             gr.update(interactive=False), 
             gr.update(interactive=False), 
             gr.update(interactive=False), 
             gr.update(interactive=False), 
             gr.update(interactive=False), # 9-13 (Buttons)
             gr.skip(), gr.skip(), gr.skip(), # 14-16 (New Res Components)
             gr.update(interactive=True), gr.update(interactive=True) # 17-18 (Stop Buttons)
         )
         
         job_dir = os.path.join(OUTPUTS_DIR, job_name)
         existing_files = glob.glob(os.path.join(job_dir, "*.mp4"))
         available_trajs = []
         for f in existing_files:
             fname = os.path.basename(f).lower()
             if fname.startswith("input_source_") and fname.endswith(".mp4") and ".depth." not in fname:
                 traj = fname.replace("input_source_", "").replace(".mp4", "")
                 if traj and traj not in available_trajs:
                     available_trajs.append(traj)
         
         last_vc = None
         last_vd = None
         
         # Pass resolution_setting
         for status, payload in render_video_gen(job_name, resolution_setting, traj_config, num_steps, low_vram):
             if status == "log":
                log_msg = payload
                
                # Create a tuple of 17 skips, but override index 7 (Log)
                # 0-6: Skip
                # 7: Log Message
                # 8-16: Skip
                update_tuple = (
                    gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), # 0-6
                    log_msg, # 7 (Log)
                    gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), # 8-12
                    gr.skip(), # 13
                    gr.skip(), gr.skip(), gr.skip(), # 14-16
                    gr.skip(), gr.skip() # 17-18
                )
                
                if progress: progress(None, desc="Rendering Video...")
                yield update_tuple
             
             elif status == "video_ready":
                 traj, vc, vd = payload
                 available_trajs.append(traj)
                 last_vc = None
                 last_vd = None
                 
                 # --- SMART UPDATE LOGIC (History) ---
                 if len(available_trajs) == 1:
                     # First video: Force update player
                     sel_update = gr.update(choices=available_trajs, value=traj)
                     vid_c_update = vc
                     vid_d_update = vd
                 else:
                     # Subsequent videos: Update list only, DO NOT RELOAD PLAYER
                     sel_update = gr.update(choices=available_trajs)
                     vid_c_update = gr.skip()
                     vid_d_update = gr.skip()
                 
                 res = list(load_job_details_by_name(job_name))
                 # Index 3 is Dropdown, 5 is Color Video, 6 is Depth Video
                 res[3] = sel_update
                 res[5] = vid_c_update
                 res[6] = vid_d_update
                 res[7] = f"[Rendering] Rendered {traj}!"
                 
                 yield tuple(res)
                 
             elif status == "done":
                 final_log, data = payload
                 msg, vc, vd = data
                 
                 if job_name in running_tasks: del running_tasks[job_name]
                 
                 res = list(load_job_details_by_name(job_name))
                 res[7] = final_log + "\n[Done] " + msg
                 
                 job_dir = os.path.join(OUTPUTS_DIR, job_name)
                 meta = load_metadata(job_dir)
                 sel = meta.get("rendered_trajectories", [])
                 
                 # Updating this dropdown triggers the client-side load event
                 res[3] = gr.update(choices=sel, value=sel[-1] if sel else None)
                 
                 yield tuple(res)
                 return
                 
             elif status == "error":
                 if job_name in running_tasks: del running_tasks[job_name]
                 yield return_refresh(payload)
                 return

    except Exception as e:
        traceback.print_exc()
        if job_name in running_tasks: del running_tasks[job_name]
        yield return_refresh(f"Error: {e}")

def load_job_details_by_name(job_name):
    """Load job details by job name."""
    
    # GLOBAL LOCK CHECK for Buttons
    busy_msg = ""
    lock_ui = False
    
    if is_system_busy():
        lock_ui = True
        if job_name in running_tasks:
            busy_msg = f"⚠️ BUSY: {running_tasks[job_name]}"
        else:
            busy_msg = f"⚠️ SYSTEM BUSY: {get_busy_message()}"
    
    # Defaults
    zip_btn_upd = gr.update(interactive=True)
    run_btn_upd = gr.update(interactive=True)

    # Empty/Error Return (Must match 19 items: 17 original + 2 stop buttons)
    empty_ret = (
        "**No job selected**", None, None, gr.update(choices=[], value=None), None, None, None, "Select a job.", 
        gr.update(visible=False), gr.update(interactive=False), gr.update(interactive=False), 
        gr.update(interactive=False), gr.update(interactive=False), run_btn_upd,
        (0,0), gr.update(), "Output Resolution: N/A",
        gr.update(interactive=False), gr.update(interactive=False) # stop_new, stop_det
    )

    if not job_name: return empty_ret
    
    job_dir = os.path.join(OUTPUTS_DIR, job_name)
    if not os.path.exists(job_dir):
        return (
            f"**{job_name}** (not found)", None, None, gr.update(choices=[], value=None), None, None, None, "Job folder not found.", 
            gr.update(visible=False), gr.update(interactive=False), gr.update(interactive=False), 
            gr.update(interactive=False), gr.update(interactive=False), run_btn_upd,
            (0,0), gr.update(), "Output Resolution: N/A",
            gr.update(interactive=False), gr.update(interactive=False)
        )
    
    # Find input image
    input_img = get_job_input_image(job_dir)
    
    # --- NEW LOGIC: Calculate Dimensions for History ---
    img_w, img_h = 0, 0
    if input_img and os.path.exists(input_img):
        try:
            with Image.open(input_img) as img:
                img_w, img_h = img.size
        except: pass
    
    # Generate smart choices for this specific image
    new_choices, default_choice = get_smart_resolution_choices(img_w, img_h)
    
    # Calculate initial output text using the fixed function
    out_res_txt = calc_final_resolution_text(img_w, img_h, default_choice)
    # ---------------------------------------------------

    all_files = sorted(glob.glob(os.path.join(job_dir, "*.*")))
    has_ply = any(f.lower().endswith(".ply") and "_gradio" not in f.lower() for f in all_files)
    
    # Find PLY file
    ply_file = None
    for f in all_files:
        if f.lower().endswith(".ply") and "_gradio" not in f.lower():
            ply_file = f
            break
    if ply_file:
        ply_file = convert_ply_for_gradio(ply_file)
    
    # Find Video Files
    vid_color = None
    vid_depth = None
    for f in all_files:
        fname = os.path.basename(f).lower()
        if fname == "input_video.mp4" or fname.endswith("color.mp4"): vid_color = f
        elif fname.endswith("depth.mp4"): vid_depth = f
    
    if not vid_color:
        for f in all_files:
            fname_lower = os.path.basename(f).lower()
            if fname_lower.endswith('.mp4') and 'depth' not in fname_lower:
                vid_color = f
                depth_cand = f.replace(".mp4", ".depth.mp4")
                if os.path.exists(depth_cand): vid_depth = depth_cand
                break
    
    vid_status = "Video available." if (vid_color or vid_depth) else "No video present."
    zip_file_update = gr.update(value=None, visible=False)
    
    if lock_ui:
        ply_update = gr.update(interactive=False)
        can_render = has_ply and check_cuda()
        vid_update = gr.update(interactive=False)
        zip_btn_upd = gr.update(interactive=False)
        del_btn_upd = gr.update(interactive=False)
        run_btn_upd = gr.update(interactive=False)
        status_text = busy_msg
    else:
        ply_update = gr.update(interactive=True)
        can_render = has_ply and check_cuda()
        vid_update = gr.update(interactive=can_render)
        zip_btn_upd = gr.update(interactive=True)
        del_btn_upd = gr.update(interactive=True)
        run_btn_upd = gr.update(interactive=True)
        status_text = f"Status: {vid_status}"
    
    # Selector Choices
    selector_choices = []
    for f in all_files:
        fname = os.path.basename(f).lower()
        if fname.startswith("input_source_") and fname.endswith(".mp4") and ".depth." not in fname:
            traj = fname.replace("input_source_", "").replace(".mp4", "")
            if traj and traj not in selector_choices: selector_choices.append(traj)
    
    if not selector_choices and vid_color:
        fname = os.path.basename(vid_color).lower()
        if "input_video" in fname: selector_choices = ["video"]
        elif fname == "input_source.mp4": selector_choices = ["video"]
        else:
            name = fname.replace("input_source_", "").replace(".mp4", "")
            if not name or name == "input_source": name = "video"
            selector_choices = [name]

    return (
        f"**Job: {job_name}**",   
        input_img,               
        ply_file,                
        gr.update(choices=selector_choices, value=selector_choices[-1] if selector_choices else None),
        all_files,  
        vid_color,               
        vid_depth,               
        status_text,             
        zip_file_update,         
        ply_update,              
        vid_update,              
        zip_btn_upd,             
        del_btn_upd,             
        run_btn_upd,
        # --- NEW RETURN VALUES (Must match refresh_outputs) ---
        (img_w, img_h),                                       # 14: det_img_dims (State)
        gr.update(choices=new_choices, value=default_choice), # 15: det_drp_resolution
        out_res_txt,                                          # 16: det_lbl_output_res
        gr.update(interactive=is_system_busy()),              # 17: btn_stop_new (Enabled if busy)
        gr.update(interactive=is_system_busy())               # 18: btn_stop_det (Enabled if busy)
    )

# ...

# Inside EVENTS ... 
# Consolidating execution flow...


# --- UI ---
theme = gr.themes.Ocean()
# --- ASSETS LOADING ---
def load_assets():
    css_path = os.path.join(BASE_DIR, "style.css")
    js_path = os.path.join(BASE_DIR, "script.js")
    
    css_content = ""
    js_content = ""
    
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as f:
            css_content = f.read()
            
    if os.path.exists(js_path):
        with open(js_path, "r", encoding="utf-8") as f:
            raw_js = f.read()
            js_content = f"<script>{raw_js}</script>"
            
    return css_content, js_content

app_css, app_js = load_assets()

with gr.Blocks(title="WebUI for ML-Sharp (3DGS)", delete_cache=(86400, 86400)) as demo:
    # Load config once to reduce terminal spam and redundant I/O
    global_cfg = load_config()
    
    gr.Markdown(f"# WebUI for ML-Sharp (3DGS)\n<p style='text-align: center; margin-top: -15px; color: var(--body-text-color-subdued);'>Version {CURRENT_VERSION}</p>")
    gr.HTML(f"""
        <h3 style='text-align: center;'>Implementation of SHARP: Sharp Monocular View Synthesis</h3>
    """)
    
    # Global Device Checks
    has_gsplat = check_gsplat()
    has_cuda = check_cuda() # Alias for compatibility
    
    with gr.Tabs(elem_id="main_tabs") as main_tabs:
        
        # --- TAB 1: NEW JOB ---
        with gr.Tab("New Job") as tab_new_job:
            with gr.Row():
                with gr.Column(scale=1):
                    new_input = gr.Image(label="Upload Image", type="filepath", height=400, elem_id="new_job_image")
                    
                    # --- START BUTTON immediately after image ---
                    with gr.Row():
                        btn_run = gr.Button("🚀 Start Generation", variant="primary", elem_id="btn_start_gen", size="lg", scale=20)
                        btn_stop_new = gr.Button("🛑", variant="stop", elem_id="btn_stop_gen_new", size="lg", scale=1, interactive=False)
                    
                    # Low VRAM moved out and above
                    low_vram_val = global_cfg.get("low_vram", False)
                    chk_low_vram = gr.Checkbox(label="Low VRAM Mode (FP16) - Lower Precision (Recommended for < 8GB VRAM)", value=low_vram_val, interactive=True)
                    
                    sl_focal = gr.Slider(label="Focal Length (mm) - [Not Detected, Default: 30]", minimum=10, maximum=200, step=1, value=30)
                    
                    # Load default preference
                    default_render = global_cfg.get("render_video", False) and has_gsplat
                    chk_render = gr.Checkbox(label="Generate Video Immediately (Requires gsplat (CUDA/MPS))", value=default_render, interactive=has_gsplat)
                    
                    with gr.Accordion("Video Customization", open=True, visible=default_render) as video_acc:
                        gr.Markdown("### Output Settings")
                        
                        # Resolution Info Labels
                        lbl_resolution = gr.Markdown("Input Resolution: N/A")
                        lbl_output_res = gr.Markdown("Output Resolution: N/A")
                        
                        # Resolution Dropdown (Updated Defaults)
                        drp_resolution = gr.Dropdown(
                            label="Output Resolution (Short Edge)",
                            choices=["FHD (1080p)", "960p", "HD (720p)", "640p", "SD (480p)"], 
                            value="FHD (1080p)", 
                            interactive=True
                        )
                        
                        gr.Markdown("### Trajectories & Depth")
                        
                        # Load trajectory preferences
                        traj_pref = global_cfg.get("trajectory_config", {
                            "rotate_forward": {"enabled": True, "depth": True},
                            "rotate": {"enabled": False, "depth": True},
                            "swipe": {"enabled": False, "depth": True},
                            "shake": {"enabled": False, "depth": True},
                            "dolly_in": {"enabled": False, "depth": True},
                            "dolly_out": {"enabled": False, "depth": True},
                            "dolly_in_out": {"enabled": False, "depth": True},
                            "pan_left": {"enabled": False, "depth": True},
                            "pan_right": {"enabled": False, "depth": True},
                            "pan_left_right": {"enabled": False, "depth": True}
                        })
                        
                        traj_controls = {}
                        traj_order = [
                            ("rotate_forward", "Rotate Forward"), ("rotate", "Full Rotate"), 
                            ("swipe", "Swipe"), ("shake", "Shake"),
                            ("dolly_in", "Dolly In"), ("dolly_out", "Dolly Out"), ("dolly_in_out", "Dolly In-Out"),
                            ("pan_left", "Pan Left"), ("pan_right", "Pan Right"), ("pan_left_right", "Pan L-R")
                        ]
                        for key, label in traj_order:
                            with gr.Row():
                                pref = traj_pref.get(key, {})
                                is_enabled = pref.get("enabled", False)
                                initial_depth = pref.get("depth", True) if is_enabled else False
                                
                                tc = gr.Checkbox(label=label, value=is_enabled, scale=3)
                                td = gr.Checkbox(label="Depth", value=initial_depth, interactive=is_enabled, scale=1)
                                traj_controls[key] = (tc, td)
                        
                        # Duration slider
                        steps_val = global_cfg.get("video_steps", 60)
                        initial_seconds = max(1, min(6, steps_val // 30))
                        sl_seconds = gr.Slider(label="Video Duration (seconds)", minimum=1, maximum=6, step=1, value=initial_seconds)
                        
                        # Link Interactivity
                        def update_depth_box(traj_enabled):
                            return gr.update(interactive=traj_enabled, value=False if not traj_enabled else gr.skip())
                        
                        for key, (tc, td) in traj_controls.items():
                            tc.change(fn=update_depth_box, inputs=[tc], outputs=[td])

                    if not has_gsplat:
                        gr.Markdown("*gsplat (CUDA/MPS) not detected: Video rendering disabled.*")
                    
                    # Events for Config Saving
                    chk_render.change(
                        fn=lambda v, img: (gr.update(visible=v), *update_ui_on_upload(img)) if v else (gr.update(visible=v), gr.skip(), gr.skip(), gr.skip(), gr.skip()), 
                        inputs=[chk_render, new_input], 
                        outputs=[video_acc, sl_focal, lbl_resolution, lbl_output_res, drp_resolution]
                    )
                    chk_render.change(fn=lambda v: save_config("render_video", v), inputs=[chk_render])
                    chk_low_vram.change(fn=lambda v: save_config("low_vram", v), inputs=[chk_low_vram])
                    
                    def save_traj_config_ui(*args):
                        keys = ["rotate_forward", "rotate", "swipe", "shake", "dolly_in", "dolly_out", "dolly_in_out", "pan_left", "pan_right", "pan_left_right"]
                        new_cfg = {}
                        for i, key in enumerate(keys):
                            new_cfg[key] = {"enabled": args[i*2], "depth": args[i*2+1]}
                        save_config("trajectory_config", new_cfg)

                    all_traj_inputs = []
                    for key in ["rotate_forward", "rotate", "swipe", "shake", "dolly_in", "dolly_out", "dolly_in_out", "pan_left", "pan_right", "pan_left_right"]:
                        all_traj_inputs.extend(list(traj_controls[key]))
                    
                    for comp in all_traj_inputs:
                        comp.change(fn=save_traj_config_ui, inputs=all_traj_inputs)

                    sl_seconds.change(fn=lambda v: save_config("video_steps", v * 30), inputs=[sl_seconds])
                    
                    # --- Event Handler Updated for Upload (Resolution & Focal) ---
                    def update_ui_on_upload(img_path):
                        # 1. Focal Logic
                        if not img_path:
                            f_update = gr.update(value=30, label="Focal Length (mm) - [Not Detected, Default: 30]")
                            w, h = 0, 0
                            mp = 0
                        else:
                            f_mm = engine.get_image_focal(img_path)
                            if f_mm:
                                f_update = gr.update(value=f_mm, label=f"Focal Length (mm) - [Detected: {f_mm:.1f}mm]")
                            else:
                                f_update = gr.update(value=30, label="Focal Length (mm) - [Not Detected, Default: 30]")
                            
                            w, h, mp, _ = get_img_resolution_data(img_path)
                        
                        # 2. Resolution Choices & Labels
                        if w > 0:
                            new_choices, default_choice = get_smart_resolution_choices(w, h)
                            in_res_txt = f"Input Resolution: {w}x{h} ({mp:.1f} MP)"
                            # Calculate initial output resolution based on default choice
                            out_res_txt = calc_final_resolution_text(w, h, default_choice)
                            return f_update, in_res_txt, out_res_txt, gr.update(choices=new_choices, value=default_choice)
                        else:
                            return f_update, "Input Resolution: N/A", "Output Resolution: N/A", gr.update()

                    # Apply to upload event
                    new_input.change(
                        fn=update_ui_on_upload, 
                        inputs=[new_input], 
                        outputs=[sl_focal, lbl_resolution, lbl_output_res, drp_resolution] 
                    )
                    
                    # --- NEW: Update output text when dropdown changes ---
                    def on_new_res_change(img_path, res_str):
                        w, h, _, _ = get_img_resolution_data(img_path)
                        return calc_final_resolution_text(w, h, res_str)

                    drp_resolution.change(
                        fn=on_new_res_change,
                        inputs=[new_input, drp_resolution],
                        outputs=[lbl_output_res]
                    )
                    
                    # --- Execution Log at the end of the column ---
                    new_log = gr.Textbox(label="Execution Log", lines=10, max_lines=20, autoscroll=True, elem_id="new_job_log")

                with gr.Column(scale=2):
                    # 3D Model Viewer
                    new_model_3d = gr.Model3D(
                        label="3D Gaussian Splat Preview",
                        height=500,
                        zoom_speed=0.5,
                        pan_speed=0.5,
                        interactive=False,
                        elem_id="new_model_3d"
                    )
                    gr.HTML("""
                        <div style="text-align: center; font-size: 0.9em; color: #888; margin-top: 5px;">
                            For best results, download the .ply file and edit it in 
                            <a href="https://superspl.at/editor" target="_blank" style="color: #4a90e2;">SuperSplat Editor</a>
                        </div>
                    """)
                    gr.Markdown("*Note: The 3D viewer FOV is fixed and may not accurately reflect the chosen focal length. Refer to rendered videos for accurate perspective.*", elem_id="fov_notice_new")
                    
                    with gr.Row():
                        new_vid_c = gr.Video(label="Video", interactive=False, height=300, loop=True, autoplay=True, elem_id="new_vid_color")
                        new_vid_d = gr.Video(label="Depth Video", interactive=False, height=300, loop=True, autoplay=True, elem_id="new_vid_depth")

                    with gr.Row():
                        drp_new_vid_selector = gr.Dropdown(label="Preview Video Trajectory", choices=[], interactive=True, scale=3, elem_id="new_vid_selector")
                    
                    new_result_files = gr.File(label="Generated Files", interactive=False, file_count="multiple", elem_id="new_result_files_list")

        # --- TAB 2: HISTORY ---
        with gr.Tab("Result History") as tab_history:
            with gr.Row():
                # LEFT COLUMN: Job List (Custom HTML)
                with gr.Column(scale=1):
                    gr.Markdown("### Recent Jobs")
                    # Custom HTML list with clickable rows
                    hist_list_html = gr.HTML(value=generate_job_list_html())
                    # Textbox for receiving selection and delete from JavaScript
                    # Use visible=True with CSS hiding so they exist in DOM
                    with gr.Row(visible=True, elem_classes="hidden-controls"):
                        job_selector = gr.Textbox(label="", elem_id="job_selector_input", container=False)
                        job_delete = gr.Textbox(label="", elem_id="job_delete_input", container=False)
                        file_delete = gr.Textbox(label="", elem_id="file_delete_input", container=False)
                
                # RIGHT COLUMN: Selected Job Details
                with gr.Column(scale=2):
                    selected_job_lbl = gr.Markdown("**No job selected**")
                    
                    # Preview Section - small image, large 3D viewer
                    with gr.Row():
                        with gr.Column(scale=1):
                            det_img = gr.Image(label="Original Input", interactive=False, height=200)
                        with gr.Column(scale=3):
                            det_model_3d = gr.Model3D(
                                label="3D Gaussian Splat",
                                height=400,
                                zoom_speed=0.5,
                                pan_speed=0.5,
                                interactive=False,
                                elem_id="det_model_3d"
                            )
                            gr.HTML("""
                                <div style="text-align: center; font-size: 0.9em; color: #888; margin-top: 5px;">
                                    For best results, download the .ply file and edit it in 
                                    <a href="https://superspl.at/editor" target="_blank" style="color: #4a90e2;">SuperSplat Editor</a>
                                </div>
                            """)
                    
                    # Video Preview Accordion
                    with gr.Accordion("📹 Video Preview", open=True):
                        drp_det_vid_selector = gr.Dropdown(label="Trajectory", choices=[], interactive=True)
                        with gr.Row():
                            det_vid_c = gr.Video(label="Color", interactive=False, height=300, loop=True, autoplay=True, elem_id="det_vid_color")
                            det_vid_d = gr.Video(label="Depth", interactive=False, height=300, loop=True, autoplay=True, elem_id="det_vid_depth")
                    
                    # Regeneration Options Accordion
                    with gr.Accordion("⚙️ Regeneration Options", open=False):
                        det_low_vram = gr.Checkbox(label="Low VRAM Mode (FP16) - Lower Precision (Recommended for < 8GB VRAM)", value=True, interactive=True)
                        det_focal = gr.Slider(label="Focal Length (mm) - [Override or keep from job]", minimum=10, maximum=200, step=1, value=30, interactive=True)
                        
                        # --- NEW: Hidden State for Image Dimensions & Output Label ---
                        det_img_dims = gr.State((0, 0)) # Stores (Width, Height) of selected job image
                        det_lbl_output_res = gr.Markdown("Output Resolution: N/A")
                        
                        # Resolution Dropdown (Updated Choices)
                        det_drp_resolution = gr.Dropdown(
                            label="Render Resolution",
                            choices=["Original", "FHD (1080p)", "960p", "HD (720p)", "640p", "SD (480p)"],
                            value="FHD (1080p)",
                            interactive=True
                        )

                        gr.Markdown("### Video Generation" + ("" if has_gsplat else " *(gsplat (CUDA/MPS) required)*"))
                        det_vid_seconds = gr.Slider(label="Video Duration (s)", minimum=1, maximum=6, value=2, step=1, interactive=has_gsplat)
                        
                        gr.Markdown("**Trajectories:**")
                        det_traj_controls = {}
                        with gr.Row():
                            det_traj_rf = gr.Checkbox(label="Rotate Forward", value=True, interactive=has_gsplat, scale=3)
                            det_traj_rf_d = gr.Checkbox(label="Depth", value=False, interactive=has_gsplat, scale=1)  # Interactive since RF is on by default
                            det_traj_controls["rotate_forward"] = (det_traj_rf, det_traj_rf_d)
                        with gr.Row():
                            det_traj_r = gr.Checkbox(label="Full Rotate", value=False, interactive=has_gsplat, scale=3)
                            det_traj_r_d = gr.Checkbox(label="Depth", value=False, interactive=False, scale=1)
                            det_traj_controls["rotate"] = (det_traj_r, det_traj_r_d)
                        with gr.Row():
                            det_traj_s = gr.Checkbox(label="Swipe", value=False, interactive=has_gsplat, scale=3)
                            det_traj_s_d = gr.Checkbox(label="Depth", value=False, interactive=False, scale=1)
                            det_traj_controls["swipe"] = (det_traj_s, det_traj_s_d)
                        with gr.Row():
                            det_traj_sh = gr.Checkbox(label="Shake", value=False, interactive=has_gsplat, scale=3)
                            det_traj_sh_d = gr.Checkbox(label="Depth", value=False, interactive=False, scale=1)
                            det_traj_controls["shake"] = (det_traj_sh, det_traj_sh_d)

                        # New Trajectories Row 1
                        with gr.Row():
                            det_traj_di = gr.Checkbox(label="Dolly In", value=False, interactive=has_gsplat, scale=3)
                            det_traj_di_d = gr.Checkbox(label="Depth", value=False, interactive=False, scale=1)
                            det_traj_controls["dolly_in"] = (det_traj_di, det_traj_di_d)
                        with gr.Row():
                            det_traj_do = gr.Checkbox(label="Dolly Out", value=False, interactive=has_gsplat, scale=3)
                            det_traj_do_d = gr.Checkbox(label="Depth", value=False, interactive=False, scale=1)
                            det_traj_controls["dolly_out"] = (det_traj_do, det_traj_do_d)
                        with gr.Row():
                            det_traj_dio = gr.Checkbox(label="Dolly In-Out", value=False, interactive=has_gsplat, scale=3)
                            det_traj_dio_d = gr.Checkbox(label="Depth", value=False, interactive=False, scale=1)
                            det_traj_controls["dolly_in_out"] = (det_traj_dio, det_traj_dio_d)

                        # New Trajectories Row 2
                        with gr.Row():
                            det_traj_pl = gr.Checkbox(label="Pan Left", value=False, interactive=has_gsplat, scale=3)
                            det_traj_pl_d = gr.Checkbox(label="Depth", value=False, interactive=False, scale=1)
                            det_traj_controls["pan_left"] = (det_traj_pl, det_traj_pl_d)
                        with gr.Row():
                            det_traj_pr = gr.Checkbox(label="Pan Right", value=False, interactive=has_gsplat, scale=3)
                            det_traj_pr_d = gr.Checkbox(label="Depth", value=False, interactive=False, scale=1)
                            det_traj_controls["pan_right"] = (det_traj_pr, det_traj_pr_d)
                        with gr.Row():
                            det_traj_plr = gr.Checkbox(label="Pan L-R", value=False, interactive=has_gsplat, scale=3)
                            det_traj_plr_d = gr.Checkbox(label="Depth", value=False, interactive=False, scale=1)
                            det_traj_controls["pan_left_right"] = (det_traj_plr, det_traj_plr_d)
                        
                        with gr.Row():
                            btn_gen_ply = gr.Button("⚙️ Regenerate 3DGS PLY", variant="primary", interactive=False, scale=10, elem_id="btn_gen_ply")
                            btn_gen_video = gr.Button("🎥 Generate Videos", variant="secondary", interactive=False, scale=10, elem_id="btn_gen_video")
                            btn_stop_det = gr.Button("🛑", variant="stop", elem_id="btn_stop_gen_det", size="lg", interactive=False, scale=1)
                    
                    # Files & Actions Accordion
                    with gr.Accordion("📁 Files & Actions", open=True):
                        with gr.Row():
                            btn_make_zip = gr.Button("📦 Create ZIP", variant="secondary", interactive=False, scale=1, elem_id="btn_zip_det")
                            btn_delete_job = gr.Button("🗑️ Delete Job", variant="stop", elem_id="btn_delete_job_details", interactive=False, scale=1)
                        
                        zip_output = gr.File(label="Download ZIP", interactive=False, visible=False)
                        det_files = gr.File(label="All Files", file_count="multiple", interactive=False, elem_id="det_files_list")
                    
                    # Operation Log
                    det_log = gr.Textbox(label="Operation Log", lines=8, max_lines=15, autoscroll=True, elem_id="det_job_log")

        # --- TAB 3: LICENSES & CREDITS ---
        with gr.Tab("Licenses & Credits"):
            gr.Markdown("### Credits & Documentation")
            
            def get_fs_file_content(path_parts):
                target_path = os.path.join(BASE_DIR, *path_parts)
                if os.path.exists(target_path):
                    try:
                        with open(target_path, "r", encoding="utf-8") as f:
                            return f.read()
                    except Exception as e:
                        return f"Error reading file: {e}"
                return "File not found."

            with gr.Accordion("WebUI for ML-Sharp License", open=False):
                gr.Code(value=get_fs_file_content(["..", "LICENSE"]), language="markdown", interactive=False)

            with gr.Accordion("ML-Sharp Guide (README)", open=False):
                gr.Markdown(get_fs_file_content(["ml-sharp", "README.md"]))



            with gr.Accordion("Third-Party Credits (gsplat MPS-Lite)", open=False):
                gr.Markdown(get_fs_file_content(["THIRD_PARTY_NOTICES.md"]))

            with gr.Accordion("Software License (ML-Sharp)", open=False):
                gr.Code(value=get_fs_file_content(["ml-sharp", "LICENSE"]), language="markdown", interactive=False)
            
            with gr.Accordion("Model License (ML-Sharp)", open=False):
                gr.Code(value=get_fs_file_content(["ml-sharp", "LICENSE_MODEL"]), language="markdown", interactive=False)
                
            with gr.Accordion("gsplat License (Apache 2.0)", open=False):
                gr.Code(value=get_fs_file_content(["licenses", "gsplat_LICENSE"]), language="markdown", interactive=False)

            with gr.Accordion("Acknowledgements & Credits", open=False):
                gr.Markdown(get_fs_file_content(["ml-sharp", "ACKNOWLEDGEMENTS"]))

        # --- TAB 4: APP GUIDE ---
        with gr.Tab("App Guide"):
            def get_project_readme():
                target_path = os.path.join(BASE_DIR, "..", "README.md")
                if os.path.exists(target_path):
                    try:
                        with open(target_path, "r", encoding="utf-8") as f:
                            return f.read()
                    except Exception as e:
                        return f"Error reading README.md: {e}"
                return "README.md not found in the project root."
            
            gr.Markdown(get_project_readme())

    
    # State for current job
    current_job_state = gr.State("")
    
    # Stores the last job generated in this session, independent of history selection.
    last_generated_job = gr.State("")
    
    
    # EVENTS
    
    # 3. Job Selection

    # The output list must match load_job_details_by_name returns: 
    # Common output list for all refresh actions
    refresh_outputs = [
        selected_job_lbl,       # 0
        det_img,                # 1
        det_model_3d,           # 2
        drp_det_vid_selector,   # 3
        det_files,              # 4
        det_vid_c,              # 5
        det_vid_d,              # 6
        det_log,                # 7
        zip_output,             # 8
        btn_gen_ply,            # 9
        btn_gen_video,          # 10
        btn_make_zip,           # 11
        btn_delete_job,         # 12
        btn_run,                # 13
        # --- NEW COMPONENTS ---
        det_img_dims,           # 14 (State)
        det_drp_resolution,     # 15 (Dropdown)
        det_lbl_output_res,     # 16 (Label)
        btn_stop_new,           # 17
        btn_stop_det            # 18
    ]

    # --- NEW: History Resolution Change Event ---
    def on_det_res_change(dims, res_str):
        w, h = dims # Retrieve from State
        return calc_final_resolution_text(w, h, res_str)

    det_drp_resolution.change(
        fn=on_det_res_change,
        inputs=[det_img_dims, det_drp_resolution],
        outputs=[det_lbl_output_res]
    )

    # 4. Video Selector Events
    def on_trajectory_change(job_name, trajectory):
        print(f"[DEBUG] on_trajectory_change: job='{job_name}', traj='{trajectory}'")
        if not job_name or not trajectory:
             return gr.update(value=None), gr.update(value=None)
        
        job_dir = os.path.join(OUTPUTS_DIR, job_name)
        
        # Try new format first: input_source_{trajectory}.mp4
        color = os.path.join(job_dir, f"input_source_{trajectory}.mp4")
        depth = os.path.join(job_dir, f"input_source_{trajectory}.depth.mp4")
        
        # Fallback logic...
        if not os.path.exists(color):
            legacy_color = os.path.join(job_dir, "input_video.mp4")
            if os.path.exists(legacy_color):
                color = legacy_color
                depth = os.path.join(job_dir, "input_video.depth.mp4")
            else:
                all_mp4 = glob.glob(os.path.join(job_dir, "*.mp4"))
                for f in all_mp4:
                    fname = os.path.basename(f).lower()
                    if "depth" not in fname:
                        color = f
                        depth_cand = f.replace(".mp4", ".depth.mp4")
                        if os.path.exists(depth_cand): depth = depth_cand
                        break
        
        if not os.path.exists(color) if color else True: color = None
        if not os.path.exists(depth) if depth else True: depth = None
            
        return color, depth

    # --- FIX 1: History Tab Selector -> MUST use 'current_job_state' ---
    # This ensures it loads videos from the job you clicked in the list.
    drp_det_vid_selector.change(
        fn=on_trajectory_change,
        inputs=[current_job_state, drp_det_vid_selector], 
        outputs=[det_vid_c, det_vid_d]
    )

    # --- FIX 2: New Job Tab Selector -> MUST use 'last_generated_job' ---
    # This ensures it loads videos from the newly created job.
    drp_new_vid_selector.change(
        fn=on_trajectory_change,
        inputs=[last_generated_job, drp_new_vid_selector], 
        outputs=[new_vid_c, new_vid_d]
    )

    # History tab trajectory depth toggle (same logic as New Job)
    def det_update_depth_box(traj_enabled):
        return gr.update(interactive=traj_enabled, value=False if not traj_enabled else gr.skip())
    
    for key, (tc, td) in det_traj_controls.items():
        tc.change(fn=det_update_depth_box, inputs=[tc], outputs=[td])

    # 3. Job Selection
    # The output list must match load_job_details_by_name returns (17 items) + state = 18
    def on_job_selected(job_name):
        print(f"[DEBUG] on_job_selected called with: '{job_name}'")
        
        if not job_name:
             # Just return defaults
             res = load_job_details_by_name("")
             yield res + ("",)
             return

        # --- PHASE 1: IMMEDIATE UI CLEAR (Visual Reset) ---
        # Yields empty values to clear the screen instantly while loading
        yield (
            "**Loading...**",          # 0: Label
            None,                      # 1: Image (Clear)
            None,                      # 2: 3D Model (Clear)
            gr.update(choices=[], value=None), # 3: Selector
            None,                      # 4: Files
            None,                      # 5: Vid C (Clear)
            None,                      # 6: Vid D (Clear)
            "Loading data...",         # 7: Log
            gr.update(visible=False),  # 8: Zip
            gr.update(interactive=False), # 9: Btn PLY
            gr.update(interactive=False), # 10: Btn Video
            gr.update(interactive=False), # 11: Btn Zip
            gr.update(interactive=False), # 12: Btn Delete
            gr.update(interactive=False), # 13: Btn Run
            (0,0),                        # 14: Dims State
            gr.update(),                  # 15: Dropdown
            "...",                        # 16: Label
            gr.update(interactive=False), # 17: Stop New
            gr.update(interactive=False), # 18: Stop Det
            job_name                      # 19: Current Job State
        )
        
        # --- PHASE 2: LOAD DATA ---
        result_tuple = load_job_details_by_name(job_name)
        yield result_tuple + (job_name,)

    # Connects the job selection click to the loading function
    job_selector.change(
        fn=on_job_selected,
        inputs=[job_selector],
        outputs=refresh_outputs + [current_job_state]
    )
    
    # 4. Job Execution - GENERATOR with Global Lock
    def predict_and_refresh(image_path, do_render_video, resolution_setting, low_vram, seconds, f_mm, current_job_name, *traj_args, progress=gr.Progress()):
        # UPDATED OUTPUTS: 13 items (added last_generated_job at the end)
        
        if is_system_busy():
            yield ("⚠️ SYSTEM BUSY", gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip())
            return
        
        # --- Reconstruct Trajectory Configuration ---
        keys = ["rotate_forward", "rotate", "swipe", "shake", "dolly_in", "dolly_out", "dolly_in_out", "pan_left", "pan_right", "pan_left_right"]
        traj_config = {}
        
        for i, key in enumerate(keys):
            traj_config[key] = {
                "enabled": traj_args[i*2], 
                "depth": traj_args[i*2+1]
            }
        
        num_steps = seconds * 30
                
        # Set Lock
        new_job_task = "Training New Job..."
        task_key = "NEW_JOB_PENDING"
        running_tasks[task_key] = new_job_task
        
        try:
            current_job_stop_event.clear()
            video_status = "Enabled" if do_render_video else "Disabled"
            log_prefix = f"[Training] (Video: {video_status})\n"
            
            # --- JOB NAME GENERATION ---
            ts = int(time.time())
            original_name = os.path.basename(image_path)
            safe_name, ext = os.path.splitext(original_name)
            job_name = f"{safe_name}_{ts}"
            
            # Initial Yield: Update BOTH states (History + New Job Memory)
            yield (
                log_prefix + "Starting training... (System Locked)", 
                gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), 
                job_name,  # 7: current_job_state (History)
                gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False),
                job_name,   # 12: last_generated_job (NEW MEMORY)
                gr.update(interactive=True), # 13: Stop button NEW
                gr.update(interactive=True)  # 14: Stop button DET
            )
            
            last_log_yield = 0
            log_update_interval = 0.5
            current_log_cache = ""
            available_trajs = []
            last_vc = None
            last_vd = None
            
            for status, payload in predict(image_path, do_render_video, resolution_setting, traj_config=traj_config, num_steps=num_steps, low_vram=low_vram, f_mm=f_mm):
                
                extracted_job_name = gr.skip()
                if status == "ply_ready":
                    ply_path = payload[0]
                    extracted_job_name = os.path.basename(os.path.dirname(ply_path))
                    job_name = extracted_job_name

                if status == "log":
                    current_log_cache = payload
                    now = time.time()
                    if now - last_log_yield >= log_update_interval:
                        last_log_yield = now
                        yield (
                            log_prefix + current_log_cache, 
                            gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), 
                            extracted_job_name, 
                            gr.skip(), gr.skip(), gr.skip(), gr.skip(),
                            extracted_job_name, # Update New Job Memory
                            gr.skip(), gr.skip() # Stop buttons
                        )
                
                elif status == "ply_ready":
                    ply_path, _, _ = payload
                    ply_view = convert_ply_for_gradio(ply_path)
                    
                    # Persist the status so JS can detect it later even after new logs arrive
                    log_prefix += "[50%] PLY Generated! Rendering videos...\n"
                    
                    yield (
                        log_prefix + current_log_cache,
                        ply_view,
                        gr.skip(), gr.skip(), gr.skip(),
                        [ply_path],
                        gr.skip(), 
                        job_name, 
                        gr.skip(), gr.skip(), gr.skip(), gr.skip(),
                        job_name, # Update New Job Memory
                        gr.skip(), gr.skip() # Stop buttons
                    )
                    
                elif status == "video_ready":
                    traj, vc, vd = payload
                    available_trajs.append(traj)
                    
                    # Yield video results incrementally
                    if len(available_trajs) == 1:
                        # Auto-load the first video in the player
                        sel_update = gr.update(choices=available_trajs, value=traj)
                        vid_c_update = gr.update(value=vc)
                        vid_d_update = gr.update(value=vd)
                    else:
                        # For subsequent videos, only update the list to avoid overloading
                        sel_update = gr.update(choices=available_trajs) 
                        vid_c_update = gr.skip()
                        vid_d_update = gr.skip()

                    yield (
                        log_prefix + current_log_cache,
                        gr.skip(), 
                        sel_update, vid_c_update, vid_d_update,
                        gr.skip(), gr.skip(), 
                        job_name,
                        gr.skip(), gr.skip(), gr.skip(), gr.skip(),
                        job_name, # Update New Job Memory
                        gr.skip(), gr.skip() # Stop buttons
                    )
                    
                elif status == "rendering":
                    traj, depth = payload
                    yield (
                        log_prefix + f"[70%] Rendering {traj}...\n" + current_log_cache,
                        gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), 
                        job_name if 'job_name' in locals() else gr.skip(),
                        gr.skip(), gr.skip(), gr.skip(), gr.skip(),
                        job_name if 'job_name' in locals() else gr.skip(), # Update New Job Memory
                        gr.skip(), gr.skip() # Stop buttons
                    )
                    
                elif status == "done":
                    final_log, result_data = payload
                    msg, _, vid_c, vid_d, job_dir = result_data
                    
                    ply_file = None
                    files_found = []
                    if job_dir and os.path.exists(job_dir):
                            raw_files = glob.glob(os.path.join(job_dir, "*.*"))
                            for f in raw_files:
                                if "_gradio" in f.lower(): continue
                                if f.lower().endswith('.ply'): ply_file = f
                                if f.lower().endswith(('.ply', '.mp4')): files_found.append(f)
                    
                    if ply_file: ply_file = convert_ply_for_gradio(ply_file)
                    
                    selector_choices = []
                    if job_dir:
                            meta = load_metadata(job_dir)
                            selector_choices = meta.get("rendered_trajectories", [])
                    
                    # FALLBACK: If metadata is empty or not yet updated, use the accumulated list
                    if not selector_choices and available_trajs:
                        selector_choices = available_trajs
                    
                    final_job_name = os.path.basename(job_dir) if job_dir else gr.skip()

                    # The dropdown update (index 2 below) will trigger the load safely.
                    yield (
                        final_log + "\n" + msg,
                        ply_file,
                        gr.update(choices=selector_choices, value=selector_choices[0] if selector_choices else None),
                        gr.skip(), # <--- FIX: vid_c (Was vid_c)
                        gr.skip(), # <--- FIX: vid_d (Was vid_d)
                        files_found,
                        generate_job_list_html(),
                        final_job_name,
                        gr.update(interactive=True),
                        gr.update(interactive=True) if job_dir else gr.skip(), 
                        gr.update(interactive=True) if job_dir else gr.skip(), 
                        gr.update(interactive=True) if (job_dir and check_cuda()) else gr.skip(),
                        final_job_name,
                        gr.update(interactive=False), # btn_stop_new (Disable)
                        gr.update(interactive=False)  # btn_stop_det (Disable)
                    )
                elif status == "error":
                    yield (
                        payload, gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), 
                        gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True),
                        gr.skip(), 
                        gr.update(interactive=False), gr.update(interactive=False)
                    )

        except Exception as e:
            import traceback
            traceback.print_exc()
            yield (f"Error: {e}", gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.update(interactive=True), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.update(interactive=False), gr.update(interactive=False))

        finally:
                if task_key in running_tasks: del running_tasks[task_key]

    # 4. Buttons
    btn_run.click(
        predict_and_refresh, 
        inputs=[new_input, chk_render, drp_resolution, chk_low_vram, sl_seconds, sl_focal, current_job_state] + all_traj_inputs, 
        outputs=[
             new_log, 
             new_model_3d,
             drp_new_vid_selector,
             new_vid_c, 
             new_vid_d,
             new_result_files, 
             hist_list_html,
             current_job_state, # Update history selection context
             btn_run,
             btn_make_zip, 
             btn_gen_ply, 
             btn_gen_video,
             last_generated_job,
             btn_stop_new, # Added
             btn_stop_det  # Added
        ],
        show_progress="hidden"
    )

    # Wire STOP Buttons
    btn_stop_new.click(fn=stop_generation, outputs=[new_log])
    btn_stop_det.click(fn=stop_generation, outputs=[det_log])
    
    # ZIP Generation
    btn_make_zip.click(
        fn=zip_job,
        inputs=[current_job_state],
        outputs=refresh_outputs,
        show_progress="hidden"
    )
    
    btn_gen_ply.click(
        fn=retry_job_ply,
        inputs=[current_job_state, det_low_vram],
        outputs=refresh_outputs,
        show_progress="hidden"
    )
    
    # History tab trajectory inputs - ALL 10 trajectories
    det_traj_inputs = [
        det_traj_rf, det_traj_rf_d,     # rotate_forward
        det_traj_r, det_traj_r_d,       # rotate
        det_traj_s, det_traj_s_d,       # swipe
        det_traj_sh, det_traj_sh_d,     # shake
        det_traj_di, det_traj_di_d,     # dolly_in
        det_traj_do, det_traj_do_d,     # dolly_out
        det_traj_dio, det_traj_dio_d,   # dolly_in_out
        det_traj_pl, det_traj_pl_d,     # pan_left
        det_traj_pr, det_traj_pr_d,     # pan_right
        det_traj_plr, det_traj_plr_d    # pan_left_right
    ]
    
    btn_gen_video.click(
        fn=regen_video_action,
        inputs=[current_job_state, det_drp_resolution, det_low_vram, det_vid_seconds] + det_traj_inputs,
        outputs=refresh_outputs,
        show_progress="hidden"
    )
    
    # 6. Delete functionality
    def delete_job_action(job_name):
        if not job_name:
             # Match History tab requirements: 1 + 19 (refresh_outputs) + 1 (state) = 21 items
             return (gr.update(),) * 20 + ("",)
            
        job_dir = os.path.join(OUTPUTS_DIR, job_name)
        if os.path.exists(job_dir):
            try:
                shutil.rmtree(job_dir)
                msg = f"Deleted {job_name}"
            except Exception as e:
                msg = f"Error deleting: {e}"
        else:
            msg = "Job not found"
        
        new_html = generate_job_list_html()
        default_state = list(load_job_details_by_name(""))
        default_state[7] = msg
        
        # 1 (html) + 19 (details) + 1 (state) = 21 items total
        return (new_html,) + tuple(default_state) + ("",)

    btn_delete_job.click(
        fn=None,
        inputs=[],
        outputs=[],
        js="triggerDeleteCurrentJob"  # Trigger delete via JS modal
    )
    
    # Also wire up inline delete from X button (triggered via job_delete textbox)
    job_delete.change(
        delete_job_action,
        inputs=[job_delete],
        outputs=[hist_list_html] + refresh_outputs + [current_job_state] # 1 + 19 + 1 = 21
    )
    
    # 7. Single File Deletion
    def delete_single_file(job_name, file_name_raw):
        print(f"DEBUG: delete_single_file called with job='{job_name}'")
        
        # helper for return
        def return_refresh(log_msg=""):
            res = list(load_job_details_by_name(job_name))
            if log_msg: res[7] = log_msg
            return tuple(res)
            
        if not job_name or not file_name_raw:
            return return_refresh("Error: No file specified")
        
        if is_system_busy():
            return return_refresh(get_busy_message())
            
        file_name_clean = file_name_raw.replace('\n', '').replace('\r', '').strip()
        base_name = os.path.basename(file_name_clean)
        
        job_dir = os.path.join(OUTPUTS_DIR, job_name)
        file_path = os.path.join(job_dir, base_name)
        
        # Fuzzy match (simplified for this update)
        if not os.path.exists(file_path):
             valid_files = glob.glob(os.path.join(job_dir, "*"))
             for f in sorted(valid_files, key=len, reverse=True):
                 if base_name.startswith(os.path.basename(f)):
                     file_path = f
                     break

        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                msg = f"Deleted file: {base_name}"
            else:
                msg = f"File not found: {base_name}"
        except Exception as e:
            msg = f"Error deleting file: {e}"
            
        return return_refresh(msg)

    file_delete.change(
        delete_single_file,
        inputs=[job_selector, file_delete],
        outputs=refresh_outputs
    )
    
    # --- NEW: Sync "New Job" Tab on Select ---
    # --- NEW: Sync "New Job" Tab on Select (Generator) ---
    def sync_new_job_from_state(job_name):
        """Refreshes the New Job tab components. Uses yield to Force-Reset the 3D viewer."""
        if not job_name:
            # Yield empty/defaults
            yield (
                None, 
                None, 
                gr.update(choices=[], value=None), 
                None, 
                None, 
                None
            )
            return
        
        # 1. Load Data
        details = load_job_details_by_name(job_name)
        
        img = details[1]
        ply = details[2]
        sel = details[3]
        files = details[4]
        
        # 2. Resolve Videos Manually (Fix for missing 'change' event)
        vid_c = None
        vid_d = None
        current_traj = sel.get('value') if isinstance(sel, dict) else None
        
        if current_traj:
            vid_c, vid_d = on_trajectory_change(job_name, current_traj)
            
        # 3. YIELD 1: RESET 
        yield (
            img, 
            None,
            sel, 
            files, 
            vid_c, 
            vid_d
        )
        
        # 4. YIELD 2: LOAD 
        yield (
            img, 
            ply, 
            sel, 
            files, 
            vid_c, 
            vid_d
        )

    tab_new_job.select(
        fn=sync_new_job_from_state,
        inputs=[last_generated_job],
        outputs=[new_input, new_model_3d, drp_new_vid_selector, new_result_files, new_vid_c, new_vid_d]
    )

    # --- NEW: Reset History Tab on Entry (TOTAL WIPE) ---
    def reset_history_view():
        """
        Total Reset when entering History Tab.
        1. Unloads 3D Model & Videos (Frees VRAM).
        2. Clears Job State.
        3. Resets the Job Selector trigger.
        4. Regenerates the HTML list with a TIMESTAMP to force a visual redraw.
        """
        # TRUCCO: Aggiungiamo un commento invisibile con l'orario.
        # Questo costringe Gradio a ridisegnare la lista, cancellando l'evidenziazione grigia.
        html = generate_job_list_html() + f""
        
        return (
            "**No job selected**",             # 0: Label
            None,                              # 1: Image (Clear)
            None,                              # 2: 3D Model (Clear)
            gr.update(choices=[], value=None), # 3: Selector (Clear)
            None,                              # 4: Files
            None,                              # 5: Vid C (Clear)
            None,                              # 6: Vid D (Clear)
            "Select a job from the list.",     # 7: Log
            gr.update(visible=False),          # 8: Zip
            gr.update(interactive=False),      # 9: Btn PLY
            gr.update(interactive=False),      # 10: Btn Video
            gr.update(interactive=False),      # 11: Btn Zip
            gr.update(interactive=False),      # 12: Btn Delete
            gr.update(interactive=False),      # 13: Btn Run
            (0,0),                             # 14: Dims State
            gr.update(),                       # 15: Res Dropdown
            "Output Resolution: N/A",          # 16: Res Label
            gr.update(interactive=False),      # 17: btn_stop_new
            gr.update(interactive=False),      # 18: btn_stop_det
            "",                                # 19: Current Job State (Empty)
            gr.update(value=None),             # 20: Job Selector (Force None)
            html                               # 21: History List HTML
        )

    tab_history.select(
        fn=reset_history_view,
        inputs=[],
        # Outputs must match the return tuple exactly (20 items)
        outputs=refresh_outputs + [current_job_state, job_selector, hist_list_html],
        js="resetHistoryUI"
    )

    # Initial Load - Triggered at the end
    # Use a Timer to trigger load to avoid race conditions
    timer = gr.Timer(value=1.0, active=True)
    
    def initial_load():
        html = generate_job_list_html()
        # libs = get_input_library_items()
        return gr.update(value=html), gr.Timer(active=False)
        
    timer.tick(initial_load, outputs=[hist_list_html, timer])

if __name__ == "__main__":
    native_allowed = [BASE_DIR, OUTPUTS_DIR]
    demo.launch(server_name="127.0.0.1", allowed_paths=native_allowed, theme=theme, css=app_css, head=app_js)