# WebUI for ML-Sharp (3DGS)

[![GitHub Sponsor](https://img.shields.io/badge/Sponsor-GitHub-ea4aaa?style=for-the-badge&logo=github-sponsors)](https://github.com/sponsors/francescofugazzi)
[![Buy Me A Coffee](https://img.shields.io/badge/Buy%20Me%20A%20Coffee-ffdd00?style=for-the-badge&logo=buy-me-a-coffee&logoColor=black)](https://www.buymeacoffee.com/franzipol)

A seamless Pinokio-ready Web UI for **Apple's ML-Sharp**, allowing you to generate 3D Gaussian Splatting (3DGS) models from a single image with high efficiency.

## 🚀 Features

- **Ultra-Fast 3DGS Generation**: High-speed production of quality 3D Gaussian Splatting (3DGS) models from a single image.
- **Single Click Install**: Fully automated dependency management via Pinokio.
- **Cross-Platform Rendering**: Support for high-performance video rendering on both **NVIDIA (CUDA)** and **Apple Silicon (MPS)**.
- **Low VRAM Support**: Optimized PLY generation modes for systems with limited GPU memory.
- **Stable GPU Workflow**: Optimized subprocess-based rendering for reliable memory release and maximum stability.
- **Flexible Resolution**: Dynamic resolution selector with predefined presets for optimal quality/speed balance.
- **Intuitive Web UI**:
  - **New Job**: Upload an image and get your 3D assets immediately.
  - **Result History**: Manage, download, and review your previous generations.
  - **Graceful Cancellation**: Instant "Stop" support to interrupt long-running jobs and release resources immediately.
- **Advanced Camera Control**: Includes 10 different camera trajectories (Orbit, Pan, Dolly, etc.).
- **Smart Resource Management**: Automatic GPU memory cleanup and cache management to keep your installation lean.

## 📦 Installation

1. Open **Pinokio**.
2. Click on **Discover** or **Download**.
3. Paste the URL of this repository.
4. Click **Install**.
5. Once finished, click **Start**.

## 🛠 Usage

1. **Upload**: Drag and drop an image into the "New Job" tab.
2. **Configure**: Choose your desired **Video Resolution** and select "Generate Video Immediately" if you have a supported GPU (NVIDIA or Mac M-series).
3. **Run**: Click "Start Generation".
4. **Download**: Once finished, download the `.ply` model or videos from the file list.
5. **History**: Access the "Result History" tab to manage your collection.

## 🖥 Requirements

- **Pinokio**: [https://pinokio.computer](https://pinokio.computer)

### 🍏 Apple Mac (Silicon)

- **Chip**: M1, M2, M3 or newer.
- **Memory**: Minimum **16GB Unified Memory** is required for stable operation.
- **Video Rendering**: Fully supported via **MPS** (Metal Performance Shaders).
  - **Note**: Includes custom **precompiled gsplat (MPS-Lite)** wheels for instant setup without compilation.

### 🖥️ Windows / Linux PC

- **GPU**: NVIDIA GPU required for CUDA video rendering.
- **VRAM**:
  - **Recommended**: **10GB+** for seamless performance.
  - **Minimum**: **8GB** supported via **Low VRAM Mode** (FP16) and shared memory.
- **Libraries**: Includes **precompiled gsplat** wheels for standard CUDA configurations.

### ⚠️ Other

- **CPU Mode**: Supported via generic installation (slower, video rendering disabled).

## ⚖️ Licenses

This software is free for **Personal, Non-Commercial, and Research Use Only**.

- **Project License**: Modified MIT (Personal/Research Usage).
- **Apple ML-Sharp**: Subject to Apple's Research Model License and Personal Software License.
- **gsplat**: Apache 2.0.

_Please review the full license details in the "Licenses & Credits" tab within the application._

## 📜 Credits

- **SHARP (ML-Sharp)**: Sharp Monocular View Synthesis in Less Than a Second. [GitHub Repository](https://github.com/apple/ml-sharp)
- **3DGS**: 3D Gaussian Splatting for Real-Time Radiance Field Rendering.

---

Developed for Pinokio Ecosystem.
