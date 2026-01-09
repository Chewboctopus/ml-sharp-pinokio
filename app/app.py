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
from sharp_engine import MLSharpEngine

# Initialize Engine
engine = MLSharpEngine()
print("Engine Initialized.")

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

def convert_ply_for_gradio(input_path: str, output_path: str = None) -> str:
    """
    Convert a 3DGS PLY file to Gradio-compatible format.
    Gradio's Gsplat.js only supports float32 properties.
    
    Args:
        input_path: Path to the original PLY file
        output_path: Optional path for the converted file. If None, creates a _gradio.ply file.
    
    Returns:
        Path to the converted PLY file
    """
    try:
        from plyfile import PlyData, PlyElement
    except ImportError:
        print("DEBUG: plyfile not installed, skipping PLY conversion")
        return input_path
    
    if not os.path.exists(input_path):
        return input_path
        
    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_gradio{ext}"
    
    try:
        print(f"DEBUG: Converting PLY for Gradio: {input_path}")
        plydata = PlyData.read(input_path)
        vertex = plydata['vertex']
        
        # Get all property names and convert to float32
        props = vertex.data.dtype.names
        num_points = len(vertex.data)
        
        # Create new dtype with all float32
        new_dtype = [(name, 'f4') for name in props]
        new_data = np.empty(num_points, dtype=new_dtype)
        
        # Copy and convert each property
        for name in props:
            new_data[name] = vertex.data[name].astype(np.float32)
        
        # Create new PLY and save
        new_element = PlyElement.describe(new_data, 'vertex')
        PlyData([new_element], text=False).write(output_path)
        
        print(f"DEBUG: PLY converted successfully: {output_path}")
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

def predict(image_path, do_render_video, traj_config=None, num_steps=60, low_vram=False, f_mm=None, progress=gr.Progress()):
    if not image_path: 
        yield ("error", "Error: No image uploaded.")
        return

    # Queue for logs
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
            ts = int(time.time())
            original_name = os.path.basename(image_path)
            safe_name, ext = os.path.splitext(original_name)
            
            job_name = f"{safe_name}_{ts}"
            job_dir = os.path.join(OUTPUTS_DIR, job_name)
            os.makedirs(job_dir, exist_ok=True)
            
            job_input = os.path.join(job_dir, f"input_source{ext}")
            shutil.copy(image_path, job_input)
            
            print(f"START JOB (Engine): {job_name}")
            
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
            # Predict
            ply_output_path = os.path.join(job_dir, f"input_source.ply")
            gaussians, f_px, res_px = engine.predict(job_input, ply_output_path, internal_resolution=resolution, f_mm_override=f_mm)
            
            # NOTIFY PLY READY
            log_queue.put(("ply_ready", (ply_output_path, f_px, res_px)))
            
            vid_color = None
            vid_depth = None
            
            if do_render_video and check_cuda():
                local_traj_config = traj_config
                if not local_traj_config:
                     local_traj_config = {"rotate_forward": {"enabled": True, "depth": True}}
                
                enabled_trajs = [(k, v["depth"]) for k, v in local_traj_config.items() if v.get("enabled", False)]
                if not enabled_trajs:
                     enabled_trajs = [("rotate_forward", True)]
                
                vid_color_path = os.path.join(job_dir, "input_source.mp4")
                rendered_videos = [] 
                
                print(f"Engine: Rendering {len(enabled_trajs)} trajectories...")
                for traj_type, do_depth in enabled_trajs:
                    log_queue.put(("rendering", (traj_type, do_depth)))
                    print(f"Engine: Rendering {traj_type} - Depth: {do_depth}...")
                    vc, vd = engine.render_video(gaussians, f_px, res_px, vid_color_path, 
                                                 trajectory_type=traj_type, num_steps=num_steps, render_depth=do_depth)
                    vid_color = vc
                    vid_depth = vd
                    rendered_videos.append(traj_type)
                    
                    # Yield incremental video result
                    log_queue.put(("video_ready", (traj_type, vc, vd)))
                    
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                meta["rendered_trajectories"] = rendered_videos
                save_metadata(job_dir, meta)
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            meta["status"] = "completed"
            save_metadata(job_dir, meta)
            
            result_container['success'] = True
            result_container['data'] = (f"Completed: {job_name}", None, vid_color, vid_depth, job_dir)
            
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
                yield (event_type, event_data)
            else:
                msg = raw_msg
                current_log += msg + "\n"
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
             current_log += msg + "\n"
             
    root_logger.removeHandler(handler)
    
    if result_container.get("success"):
        yield ("done", (current_log, result_container['data']))
    else:
        err = result_container.get("error", "Unknown Error")
        yield ("error", current_log + "\nError: " + err)

def render_video_gen(job_name, traj_config=None, num_steps=60, low_vram=False):
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
            
            print(f"Engine (Regen): Rendering {len(enabled_trajs)} trajectories...")
            for traj_type, do_depth in enabled_trajs:
                print(f"Engine (Regen): Rendering {traj_type} ({num_steps} steps) - Depth: {do_depth}...")
                vc, vd = engine.render_video(gaussians, metadata.focal_length_px, metadata.resolution_px, 
                                            vid_color_path, trajectory_type=traj_type, num_steps=num_steps, render_depth=do_depth)
                last_vc, last_vd = vc, vd
                rendered_videos.append(traj_type)
                
                # Notify about completed video immediately
                log_queue.put(("video_ready", (traj_type, vc, vd)))
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            meta = load_metadata(job_dir)
            meta["rendered_trajectories"] = rendered_videos
            save_metadata(job_dir, meta)

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
                yield (event_type, event_data)
            else:
                current_log += raw_msg + "\n"
                yield ("log", current_log)
        except queue.Empty:
            time.sleep(0.1)
            
    while not log_queue.empty():
        raw_msg = log_queue.get()
        if isinstance(raw_msg, tuple):
            event_type, event_data = raw_msg
            yield (event_type, event_data)
        else:
            current_log += raw_msg + "\n"
        
    root_logger.removeHandler(handler)
    
    if result_container.get("success"):
        yield ("done", (current_log, result_container['data']))
    else:
        err = result_container.get("error", "Unknown Error")
        yield ("error", current_log + "\nError: " + err)

def load_job_details(evt: gr.SelectData):
    # evt.value is the caption if gallery mode is caption. But here we pass tuples (img, label)
    # If we use tuples in gallery, evt.value is the Label (job_name) if defined?
    # Gradio Gallery returns index mostly.
    
    # Retrieve job name from current list
    # In simpler mode: use index
    # For Dataframe: evt.index is [row_index, col_index]
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
        # [lbl, img, model3d, selector, files, vc, vd, log, zip_out, ply_btn, vid_btn, make_zip_btn, del_btn, run_btn]
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
    input_file = None
    candidates = glob.glob(os.path.join(job_dir, "*"))
    for f in candidates:
        if "input" in os.path.basename(f).lower() and f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
            input_file = f
            break
            
    if not input_file:
        for f in candidates:
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                input_file = f
                break
    
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
        gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(),
        "[Generating] " + running_tasks[job_name] + " (Queued)...", 
        gr.skip(), 
        gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False)
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

def regen_video_action(job_name, low_vram, seconds, *traj_args, progress=gr.Progress()):
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
         # Initial: Loading with marker for JS animation
         yield (
             gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(),
             "[Rendering] Starting Video Render...", 
             gr.skip(), 
             gr.update(interactive=False), 
             gr.update(interactive=False), 
             gr.update(interactive=False), 
             gr.update(interactive=False), 
             gr.update(interactive=False)
         )
         
         # Pre-populate with existing trajectories from job folder
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
         
         for status, payload in render_video_gen(job_name, traj_config, num_steps, low_vram):
             if status == "log":
                 res = list(load_job_details_by_name(job_name))
                 # Only update log (index 7) with marker, preserve videos
                 res[7] = "[Rendering] " + payload
                 if last_vc: res[5] = last_vc
                 if last_vd: res[6] = last_vd
                 if available_trajs:
                     res[3] = gr.update(choices=available_trajs, value=available_trajs[-1])
                 if progress: progress(None, desc="Rendering Video...")
                 yield tuple(res)
             
             elif status == "video_ready":
                 traj, vc, vd = payload
                 available_trajs.append(traj)
                 last_vc = vc
                 last_vd = vd
                 
                 res = list(load_job_details_by_name(job_name))
                 res[3] = gr.update(choices=available_trajs, value=traj)
                 res[5] = vc
                 res[6] = vd
                 res[7] = f"[Rendering] Rendered {traj}!"
                 yield tuple(res)
                 time.sleep(0.2)  # Give UI time to update
                 
             elif status == "done":
                 final_log, data = payload
                 msg, vc, vd = data
                 
                 # Clear task BEFORE final yield so buttons unlock
                 if job_name in running_tasks: del running_tasks[job_name]
                 
                 res = list(load_job_details_by_name(job_name))
                 res[7] = final_log + "\n[Done] " + msg
                 if vc: res[5] = vc
                 if vd: res[6] = vd
                 
                 job_dir = os.path.join(OUTPUTS_DIR, job_name)
                 meta = load_metadata(job_dir)
                 sel = meta.get("rendered_trajectories", [])
                 res[3] = gr.update(choices=sel, value=sel[-1] if sel else None)
                 
                 yield tuple(res)
                 return  # Exit early, task already cleared
                 
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

    if not job_name:
         # lbl, img, model3d, selector, files, vc, vd, log, zip_file, ply_btn, vid_btn, make_zip_btn, del_btn, run_btn
        return "**No job selected**", None, None, gr.update(choices=[], value=None), None, None, None, "Select a job.", gr.update(visible=False), gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False), run_btn_upd
    
    job_dir = os.path.join(OUTPUTS_DIR, job_name)
    if not os.path.exists(job_dir):
        return f"**{job_name}** (not found)", None, None, gr.update(choices=[], value=None), None, None, None, "Job folder not found.", gr.update(visible=False), gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False), run_btn_upd
    
    # Find input image
    input_img = None
    candidates = glob.glob(os.path.join(job_dir, "*"))
    for f in candidates:
         if "input" in os.path.basename(f).lower() and f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
            input_img = f
            break
    if not input_img:
        for f in candidates:
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                input_img = f
                break
    
    all_files = sorted(candidates)
    has_ply = any(f.lower().endswith(".ply") and "_gradio" not in f.lower() for f in all_files)
    
    # Find PLY file for 3D viewer
    ply_file = None
    for f in all_files:
        if f.lower().endswith(".ply") and "_gradio" not in f.lower():
            ply_file = f
            break
    
    # Convert PLY to Gradio-compatible format
    if ply_file:
        ply_file = convert_ply_for_gradio(ply_file)
        print(f"DEBUG: PLY file for viewer {job_name}: {ply_file}")
    
    vid_color = None
    vid_depth = None
    for f in all_files:
        fname = os.path.basename(f).lower()
        if fname == "input_video.mp4" or fname.endswith("color.mp4"): vid_color = f
        elif fname.endswith("depth.mp4"): vid_depth = f
    
    if not vid_color:
        # Sort to prioritize certain names or just find any mp4
        for f in all_files:
            fname_lower = os.path.basename(f).lower()
            if fname_lower.endswith('.mp4') and 'depth' not in fname_lower:
                vid_color = f
                # If we find a depth counterpart, set it
                depth_cand = f.replace(".mp4", ".depth.mp4")
                if os.path.exists(depth_cand):
                    vid_depth = depth_cand
                break
    
    vid_status = "Video available." if (vid_color or vid_depth) else "No video present."
    
    # Defaults
    zip_file_update = gr.update(value=None, visible=False)
    
    # Lock logic overrides
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
    
    # Extract selector list by scanning ACTUAL mp4 files (source of truth)
    selector_choices = []
    
    # Scan for trajectory mp4 files: input_source_{trajectory}.mp4
    for f in all_files:
        fname = os.path.basename(f).lower()
        if fname.startswith("input_source_") and fname.endswith(".mp4") and ".depth." not in fname:
            # Extract trajectory name: input_source_rotate.mp4 -> rotate
            traj = fname.replace("input_source_", "").replace(".mp4", "")
            if traj and traj not in selector_choices:
                selector_choices.append(traj)
    
    # Fallback for very old jobs with input_video.mp4 (no trajectories in filename)
    if not selector_choices and vid_color:
        fname = os.path.basename(vid_color).lower()
        if "input_video" in fname:
            # Very old format - just call it "video"
            selector_choices = ["video"]
        elif fname == "input_source.mp4":
            # Another very old format
            selector_choices = ["video"]
        else:
            # Try to extract trajectory name
            name = fname.replace("input_source_", "").replace(".mp4", "")
            if not name or name == "input_source": 
                name = "video"
            selector_choices = [name]

    return (
        f"**Job: {job_name}**",   # Label
        input_img,               # Original Input
        ply_file,                # 3D Viewer
        gr.update(choices=selector_choices, value=selector_choices[-1] if selector_choices else None), # New selector
        all_files,  # File links - all files, JS handles hiding delete buttons for protected files
        vid_color,               # Video
        vid_depth,               # Depth
        status_text,             # Log
        zip_file_update,         # ZIP Download box
        ply_update,              # btn_gen_ply
        vid_update,              # btn_gen_video
        zip_btn_upd,             # btn_make_zip
        del_btn_upd,             # btn_delete_job
        run_btn_upd              # btn_run
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
    
    gr.Markdown("# WebUI for ML-Sharp (3DGS)")
    gr.HTML("<h3 style='text-align: center;'>Implementation of SHARP: Sharp Monocular View Synthesis in Less Than a Second</h3>")
    
    # Global Device Checks
    has_gsplat = check_gsplat()
    has_cuda = check_cuda() # Alias for compatibility
    
    with gr.Tabs(elem_id="main_tabs") as main_tabs:
        
        # --- TAB 1: NEW JOB ---
        with gr.Tab("New Job"):
            with gr.Row():
                with gr.Column(scale=1):
                    new_input = gr.Image(label="Upload Image", type="filepath")
                    
                    # Low VRAM moved out and above
                    low_vram_val = global_cfg.get("low_vram", False)
                    chk_low_vram = gr.Checkbox(label="Low VRAM Mode (FP16) - Lower Precision (Recommended for < 8GB VRAM)", value=low_vram_val, interactive=True)
                    
                    sl_focal = gr.Slider(label="Focal Length (mm) - [Not Detected, Default: 30]", minimum=10, maximum=200, step=1, value=30)
                    
                    # Load default preference, but force False if no gsplat
                    default_render = global_cfg.get("render_video", False) and has_gsplat
                    chk_render = gr.Checkbox(label="Generate Video Immediately (Requires gsplat (CUDA/MPS))", value=default_render, interactive=has_gsplat)
                    
                    with gr.Accordion("Video Customization", open=True, visible=default_render) as video_acc:
                        # (Removed internal resolution notice)
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
                                # Force False for depth if not enabled, regardless of what's in config
                                initial_depth = pref.get("depth", True) if is_enabled else False
                                
                                tc = gr.Checkbox(label=label, value=is_enabled, scale=3)
                                td = gr.Checkbox(label="Depth", value=initial_depth, interactive=is_enabled, scale=1)
                                traj_controls[key] = (tc, td)
                        
                        # Duration in seconds (1-6)
                        steps_val = global_cfg.get("video_steps", 60)
                        initial_seconds = max(1, min(6, steps_val // 30))
                        sl_seconds = gr.Slider(label="Video Duration (seconds)", minimum=1, maximum=6, step=1, value=initial_seconds)
                        
                        # Link Interactivity & Values
                        def update_depth_box(traj_enabled):
                            return gr.update(interactive=traj_enabled, value=False if not traj_enabled else gr.skip())
                        
                        for key, (tc, td) in traj_controls.items():
                            tc.change(fn=update_depth_box, inputs=[tc], outputs=[td])

                    if not has_gsplat:
                        gr.Markdown("*gsplat (CUDA/MPS) not detected: Video rendering disabled.*")
                    
                    # Save preference on change
                    chk_render.change(fn=lambda v: gr.update(visible=v), inputs=[chk_render], outputs=[video_acc])
                    chk_render.change(fn=lambda v: save_config("render_video", v), inputs=[chk_render])
                    chk_low_vram.change(fn=lambda v: save_config("low_vram", v), inputs=[chk_low_vram])
                    
                    # Helper to save trajectory config
                    def save_traj_config_ui(*args):
                        # args will be [rf_c, rf_d, r_c, r_d, s_c, s_d, sh_c, sh_d]
                        # We need to map them back
                        keys = ["rotate_forward", "rotate", "swipe", "shake", "dolly_in", "dolly_out", "dolly_in_out", "pan_left", "pan_right", "pan_left_right"]
                        new_cfg = {}
                        for i, key in enumerate(keys):
                            new_cfg[key] = {"enabled": args[i*2], "depth": args[i*2+1]}
                        
                        # Ensure at least one is enabled
                        if not any(v["enabled"] for v in new_cfg.values()):
                             # Revert or warn? For now we just save and handle in predict
                             pass
                        
                        save_config("trajectory_config", new_cfg)

                    # Wire up all trajectory checkboxes
                    all_traj_inputs = []
                    for key in ["rotate_forward", "rotate", "swipe", "shake", "dolly_in", "dolly_out", "dolly_in_out", "pan_left", "pan_right", "pan_left_right"]:
                        all_traj_inputs.extend(list(traj_controls[key]))
                    
                    for comp in all_traj_inputs:
                        comp.change(fn=save_traj_config_ui, inputs=all_traj_inputs)

                    # Map seconds to steps (30fps) for config
                    sl_seconds.change(fn=lambda v: save_config("video_steps", v * 30), inputs=[sl_seconds])
                    
                    # Update Focal Length on image upload
                    # Update Focal Length on image upload
                    def update_focal_on_upload(img):
                        if not img: return gr.update(value=30, label="Focal Length (mm) - [Not Detected, Default: 30]")
                        f_mm = engine.get_image_focal(img)
                        if f_mm:
                             return gr.update(value=f_mm, label=f"Focal Length (mm) - [Detected: {f_mm:.1f}mm]")
                        return gr.update(value=30, label="Focal Length (mm) - [Not Detected, Default: 30]")
                    new_input.change(fn=update_focal_on_upload, inputs=[new_input], outputs=[sl_focal])
                    
                    btn_run = gr.Button("Start Generation", variant="primary", elem_id="btn_start_gen")
                    new_log = gr.Textbox(label="Execution Log", lines=10, max_lines=20, autoscroll=True, elem_id="new_job_log")

                with gr.Column(scale=2):
                    # 3D Model Viewer - Camera settings matching Apple ML-Sharp rotate_forward start
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
                        new_vid_c = gr.Video(label="Video", interactive=False, loop=True, autoplay=True, elem_id="new_vid_color")
                        new_vid_d = gr.Video(label="Depth Video", interactive=False, loop=True, autoplay=True, elem_id="new_vid_depth")

                    with gr.Row():
                        drp_new_vid_selector = gr.Dropdown(label="Preview Video Trajectory", choices=[], interactive=True, scale=3, elem_id="new_vid_selector")
                        # We don't need a button, change event will update
                    
                    new_result_files = gr.File(label="Generated Files", interactive=False, file_count="multiple", elem_id="new_result_files_list")

        # --- TAB 2: HISTORY ---
        with gr.Tab("Result History"):
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
                            btn_gen_ply = gr.Button("⚙️ Regenerate 3DGS PLY", variant="primary", interactive=False)
                            btn_gen_video = gr.Button("🎥 Generate Videos", variant="secondary", interactive=False)
                    
                    # Files & Actions Accordion
                    with gr.Accordion("📁 Files & Actions", open=True):
                        with gr.Row():
                            btn_make_zip = gr.Button("📦 Create ZIP", variant="secondary", interactive=False, scale=1)
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
    
    # State for library selection (Removed)
    # selected_lib_item = gr.State(None)
    
    # EVENTS
    
    # 3. Job Selection

    # The output list must match load_job_details_by_name returns: 
    # Common output list for all refresh actions (now Includes ZIP and RUN buttons)
    refresh_outputs = [
        selected_job_lbl, 
        det_img, 
        det_model_3d,
        drp_det_vid_selector,
        det_files, 
        det_vid_c, 
        det_vid_d, 
        det_log, 
        zip_output,
        btn_gen_ply,
        btn_gen_video,
        btn_make_zip,
        btn_delete_job,
        btn_run
    ]

    # 4. Video Selector Events
    def on_trajectory_change(job_name, trajectory):
        print(f"[DEBUG] on_trajectory_change: job='{job_name}', traj='{trajectory}'")
        if not job_name or not trajectory:
             return gr.update(value=None), gr.update(value=None)
        
        job_dir = os.path.join(OUTPUTS_DIR, job_name)
        
        # Try new format first: input_source_{trajectory}.mp4
        color = os.path.join(job_dir, f"input_source_{trajectory}.mp4")
        depth = os.path.join(job_dir, f"input_source_{trajectory}.depth.mp4")
        
        # Fallback for old formats
        if not os.path.exists(color):
            # Try legacy: input_video.mp4
            legacy_color = os.path.join(job_dir, "input_video.mp4")
            if os.path.exists(legacy_color):
                color = legacy_color
                depth = os.path.join(job_dir, "input_video.depth.mp4")
            else:
                # Try any mp4 that's not depth
                all_mp4 = glob.glob(os.path.join(job_dir, "*.mp4"))
                for f in all_mp4:
                    fname = os.path.basename(f).lower()
                    if "depth" not in fname:
                        color = f
                        # Look for matching depth
                        depth_cand = f.replace(".mp4", ".depth.mp4")
                        if os.path.exists(depth_cand):
                            depth = depth_cand
                        break
        
        if not os.path.exists(color) if color else True: 
            print(f"[DEBUG] Color video not found")
            color = None
        if not os.path.exists(depth) if depth else True: 
            print(f"[DEBUG] Depth video not found")
            depth = None
            
        return color, depth

    drp_new_vid_selector.change(
        fn=on_trajectory_change,
        inputs=[current_job_state, drp_new_vid_selector],
        outputs=[new_vid_c, new_vid_d]
    )
    drp_det_vid_selector.change(
        fn=on_trajectory_change,
        inputs=[current_job_state, drp_det_vid_selector],
        outputs=[det_vid_c, det_vid_d]
    )

    # History tab trajectory depth toggle (same logic as New Job)
    def det_update_depth_box(traj_enabled):
        return gr.update(interactive=traj_enabled, value=False if not traj_enabled else gr.skip())
    
    for key, (tc, td) in det_traj_controls.items():
        tc.change(fn=det_update_depth_box, inputs=[tc], outputs=[td])

    # 3. Job Selection
    # The output list must match load_job_details_by_name returns (11 items) + state = 12
    def on_job_selected(job_name):
        print(f"[DEBUG] on_job_selected called with: '{job_name}'")
        if not job_name:
             # Match return count: 12 + 1 = 13 (refresh_outputs has 12 items)
             # Use load_job_details_by_name("") to get correct defaults
             return load_job_details_by_name("") + ("",)
        result = load_job_details_by_name(job_name)
        return result + (job_name,)
        
    job_selector.change(
        fn=on_job_selected,
        inputs=[job_selector],
        outputs=refresh_outputs + [current_job_state] # 12 + 1 = 13
    )
    
    # 4. Job Execution - GENERATOR with Global Lock
    def predict_and_refresh(image_path, do_render_video, low_vram, seconds, f_mm, current_job_name, *traj_args):
        # traj_args is [rf_c, rf_d, r_c, r_d, s_c, s_d, sh_c, sh_d]
        if is_system_busy():
            yield ("⚠️ SYSTEM BUSY", gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip())
            return
        
        keys = ["rotate_forward", "rotate", "swipe", "shake", "dolly_in", "dolly_out", "dolly_in_out", "pan_left", "pan_right", "pan_left_right"]
        traj_config = {}
        for i, key in enumerate(keys):
            traj_config[key] = {"enabled": traj_args[i*2], "depth": traj_args[i*2+1]}
        
        num_steps = seconds * 30
            
        # Set Lock
        new_job_task = "Training New Job..."
        task_key = "NEW_JOB_PENDING"
        running_tasks[task_key] = new_job_task
        
        try:
             # Initial Yield: BUSY STATE
             # [new_log, new_model_3d, selector, new_vid_c, new_vid_d, new_result_files, hist_list, job_state, btn_run, btn_make_zip, btn_gen_ply, btn_gen_video]
             yield (
                 "[Training] Starting training... (System Locked)", 
                 gr.skip(),  # new_model_3d
                 gr.skip(),  # selector
                 gr.skip(), gr.skip(), gr.skip(), # vids, files
                 gr.skip(), # hist_list
                 gr.skip(), # job_state
                 gr.update(interactive=False), # btn_run
                 gr.update(interactive=False), # btn_make_zip
                 gr.update(interactive=False), # btn_gen_ply
                 gr.update(interactive=False)  # btn_gen_video
             )
             
             # Consume Generator with throttling
             last_log_yield = 0
             log_update_interval = 0.5  # seconds
             current_log_cache = ""
             available_trajs = []
             last_vc = None  # Track last video paths
             last_vd = None
             
             for status, payload in predict(image_path, do_render_video, traj_config=traj_config, num_steps=num_steps, low_vram=low_vram, f_mm=f_mm):
                 if status == "log":
                     current_log_cache = payload
                     now = time.time()
                     # Only yield log updates every log_update_interval seconds
                     if now - last_log_yield >= log_update_interval:
                         last_log_yield = now
                         # Use last known video paths to prevent flickering
                         vid_c_val = last_vc if last_vc else gr.skip()
                         vid_d_val = last_vd if last_vd else gr.skip()
                         yield (
                             "[Training] " + current_log_cache, 
                             gr.skip(), gr.skip(), vid_c_val, vid_d_val, gr.skip(), gr.skip(), gr.skip(), 
                             gr.skip(), gr.skip(), gr.skip(), gr.skip()
                         )
                 
                 elif status == "ply_ready":
                     ply_path, _, _ = payload
                     ply_view = convert_ply_for_gradio(ply_path)
                     yield (
                         "[50%] PLY Generated! Rendering videos...\n" + current_log_cache,
                         ply_view,
                         gr.skip(), gr.skip(), gr.skip(),
                         [ply_path],
                         gr.skip(), gr.skip(),
                         gr.skip(), gr.skip(), gr.skip(), gr.skip()
                     )
                     

                     
                 elif status == "video_ready":
                     traj, vc, vd = payload
                     # Update video selector with new trajectory
                     available_trajs.append(traj)
                     
                     # Track last video paths to prevent flickering
                     last_vc = vc
                     last_vd = vd
                     
                     # Persist the message so it doesn't disappear on next log update
                     done_msg = f"Rendered {traj}!"
                     current_log_cache = done_msg + "\n" + current_log_cache
                     
                     print(f"DEBUG: Yielding video update for {traj}: C={vc}, D={vd}")
                     yield (
                         "[Training] " + current_log_cache,
                         gr.skip(), 
                         gr.update(choices=available_trajs, value=traj), # Update selector and select newest
                         vc,  # Pass path directly like regen_video_action
                         vd,  # Pass path directly 
                         gr.skip(), # file list
                         gr.skip(), gr.skip(), 
                         gr.skip(), gr.skip(), gr.skip(), gr.skip()
                     )
                     # Brief pause to ensure frontend receives and processes the video update 
                     # before the next log update comes rushing in.
                     time.sleep(0.2)
                     
                 elif status == "rendering":
                     traj, depth = payload
                     # Use last known video paths to prevent flickering
                     vid_c_val = last_vc if last_vc else gr.skip()
                     vid_d_val = last_vd if last_vd else gr.skip()
                     yield (
                         f"[Training] [70%] Rendering {traj}...\n" + current_log_cache,
                         gr.skip(), gr.skip(), vid_c_val, vid_d_val, gr.skip(), gr.skip(), gr.skip(), 
                         gr.skip(), gr.skip(), gr.skip(), gr.skip()
                     )
                     
                 elif status == "done":
                     final_log, result_data = payload
                     msg, _, vid_c, vid_d, job_dir = result_data
                     
                     # Process Results similar to before
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
                     
                     # Final Yield: Enable all
                     yield (
                         final_log + "\n" + msg,
                         ply_file,
                         gr.update(choices=selector_choices, value=selector_choices[-1] if selector_choices else None),
                         vid_c,
                         vid_d,
                         files_found,
                         generate_job_list_html(),
                         os.path.basename(job_dir) if job_dir else gr.skip(),
                         gr.update(interactive=True), # btn_run
                         gr.update(interactive=True) if job_dir else gr.skip(), 
                         gr.update(interactive=True) if job_dir else gr.skip(), 
                         gr.update(interactive=True) if (job_dir and check_cuda()) else gr.skip()
                     )
                 elif status == "error":
                     yield (
                         payload, gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), 
                         gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True)
                     )

        except Exception as e:
            import traceback
            traceback.print_exc()
            yield (f"Error: {e}", gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.update(interactive=True), gr.skip(), gr.skip(), gr.skip())

        finally:
             if task_key in running_tasks: del running_tasks[task_key]

    # 4. Buttons
    btn_run.click(
        predict_and_refresh, 
        inputs=[new_input, chk_render, chk_low_vram, sl_seconds, sl_focal, current_job_state] + all_traj_inputs, 
        outputs=[
             new_log, 
             new_model_3d,
             drp_new_vid_selector,
             new_vid_c, 
             new_vid_d,
             new_result_files, 
             hist_list_html,
             current_job_state, # Update state after job
             btn_run,
             btn_make_zip, 
             btn_gen_ply, 
             btn_gen_video
        ],
        show_progress="hidden"
    )
    
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
        inputs=[current_job_state, det_low_vram, det_vid_seconds] + det_traj_inputs,
        outputs=refresh_outputs,
        show_progress="hidden"
    )
    
    # 6. Delete functionality
    def delete_job_action(job_name):
        if not job_name:
             # Return updates preserving current UI but clearing state if needed
             return (gr.update(),) * 14 + ("",)
            
        job_dir = os.path.join(OUTPUTS_DIR, job_name)
        if os.path.exists(job_dir):
            try:
                shutil.rmtree(job_dir)
                msg = f"Deleted {job_name}"
            except Exception as e:
                msg = f"Error deleting: {e}"
        else:
            msg = "Job not found"
        
        # Refresh HTML list and clear details
        new_html = generate_job_list_html()
        
        # Get default "empty" state for right panel
        default_state = list(load_job_details_by_name(""))
        # Index 7 is status_text/log
        default_state[7] = msg
        
        # Return new_html + default_state (14 items) + empty state string
        # Match order: hist_list_html (1) + load_job_details_by_name (14) + current_job_state (1) = 16 items
        return (new_html,) + tuple(default_state) + ("",)

    btn_delete_job.click(
        fn=None,
        inputs=[],
        outputs=[],
        js="triggerDeleteCurrentJob"  # Trigger delete via JS modal
    )
    
    # Also wire up inline delete from X button (triggered via job_delete textbox)
    # Also wire up inline delete from X button (triggered via job_delete textbox)
    job_delete.change(
        delete_job_action,
        inputs=[job_delete],
        outputs=[hist_list_html] + refresh_outputs + [current_job_state] # 1 + 12 + 1 = 14
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
    
    # 6. Input Library Logic
    
    # Select from Library -> Update State Only (User must click "Use" to confirm)
    # Library Logic Removed 


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