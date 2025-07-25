# Conveyor Belt Anomaly Detection

A deep learning-based system for real-time anomaly detection on conveyor belt video streams, featuring a ResNet-based Variational Autoencoder (VAE), flexible training/inference scripts, and a web dashboard for live monitoring and control.

This project was done to get over the needing of the labeled data, precisely the classic way of having normal VS anormal data for machine learning detection models, VAE here are only using the normal data to help detect defections in this exemple on a conveyor belt, so please tune this project based on what you want to achieve, Enjoy !

DISCLAIMER : THIS PROJECT NEEDS A MINIMUM OF A 800 CUDA CORES CAPABLE GPU OR EQUIVALENT PERFORMANCES FOR THE BEST EXPERIENCE ! PLEASE MAKE THIS INTO CONSIDERATION WHILE YOU TRYING TO IMPLEMETING THIS PROJECT INTO YOUR NEEDINGS.

To see exemples you can checkout the "output_anomalies_online" for output samples of what the model is capable of only from normal Data !
---

## Table of Contents
- [Features](#features)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Setup](#setup)
- [Usage](#usage)
  - [1. Data Extraction & Preprocessing](#1-data-extraction--preprocessing)
  - [2. Model Training](#2-model-training)
  - [3. Inference & Anomaly Detection](#3-inference--anomaly-detection)
  - [4. Web Dashboard (Linux Only)](#4-web-dashboard-linux-only)
- [Useful Commands](#useful-commands)
- [Notes on OS Compatibility](#notes-on-os-compatibility)
- [Contributing](#contributing)

---

## Features
- **ResNet-based VAE** for robust anomaly detection in video frames (This is the Model Structure)
- **Offline and real-time (RTSP) analysis**
- **Web dashboard** (Flask, Linux only) for live control, logs, and parameter tuning
- **Automatic anomaly visualization** (original, reconstruction, heatmap)
- **Highly configurable** (thresholds, sampling rates, etc.)
- **Output management**: saves flagged anomaly frames for review
- **Jupyter notebooks** for data extraction, preprocessing, and model experimentation and training if needed

---

## Project Structure

```
.
├── resnet.py                  # ResNetVAE_V2 model (standalone)
├── test_script_linux.py       # Main Linux inference/production script
├── cmds_usefull.txt           # Useful command-line examples
├── conda_env_win_configuration.yml # Conda environment (Windows, adaptable for Linux) | And mediamtx.yml (the mediamtx config file ONLY FOR LINUX)
├── output_anomalies_online/   # Output: anomaly images (not tracked)
├── training_phase/
│   ├── analyse_video_offline.py   # Offline video analysis (Windows/Linux)
│   ├── analyse_video_online.py    # RTSP/stream analysis (Windows/Linux)
│   ├── extracting-processing.ipynb # Frame extraction/preprocessing
│   ├── model_resalone.ipynb       # Model training/experiments
│   └── models/                    # Model definitions (ResNetVAE, VAEs, etc.)
├── web_support/               # Web dashboard (Linux only)
│   ├── app.py                     # Flask backend
│   ├── templates/                 # HTML templates
│   └── static/                    # CSS, assets
└── README.md                  # This file
```

---

## Requirements
- **Python 3.8+**
- **PyTorch** (see conda_env_win_configuration.yml for version)
- **torchvision, numpy, opencv-python, matplotlib, flask, tqdm, lpips**
- **Linux (recommended for production/web dashboard)**
- **Windows**: Supported for training and offline/online analysis scripts (no web dashboard)

> **Tip:** Use the provided `conda_env_win_configuration.yml` as a base. For Linux, install packages manually as needed (see script imports).

---

## Setup

1. **Clone the repository:**
   ```sh
   git clone https://github.com/yourusername/conveyor_belt_anomalies_detection_ML.git
   cd conveyor_belt_anomalies_detection_ML
   ```
2. **Create the environment:**
   - **Windows:**
     ```sh
     conda env create -f conda_env_win_configuration.yml
     conda activate <your_env_name>
     ```
   - **Linux:**
     - Use the YAML as a reference, but install packages manually to avoid Windows-specific issues:
     ```sh
     conda create -n conveyor_belt python=3.8
     conda activate conveyor_belt
     # Install packages as per the YAML and script imports
     pip install torch torchvision numpy opencv-python matplotlib flask tqdm lpips
     ```
3. **Download or train model weights:**
   - Place your trained weights as `resnet_v2_scratch_vae_best.pth` in the appropriate path (see script arguments).
4. **(Optional) Review `cmds_usefull.txt`** for more command-line examples and troubleshooting tips.

---

## Usage

### 1. Data Extraction & Preprocessing
- Use `extracting-processing.ipynb` to extract frames from videos and preprocess them for training (the video shall be used needs to be in the mp4 format).

### 2. Model Training
- Use `model_resalone.ipynb` (or your own script) to train the ResNetVAE_V2 model on your dataset.
- Save the best model as `resnet_v2_scratch_vae_best.pth`.

### 3. Inference & Anomaly Detection
- **Offline (video file):** [Tip you can change the threshold to aline with your expectations and tune your model sensivity]
  ```sh
  python training_phase/analyse_video_offline.py --video_path "path/to/video.mp4" --model_path "path/to/resnet_v2_scratch_vae_best.pth" --threshold 0.14 --use_lpips --frame_sampling_rate 3 --latent_dim 512 --output_anomaly_dir "./anomalies_lpips_test2"
  ```
- **Online (RTSP stream):**

  **Lunch the Mediamtx and the ffmpeg to share the stream:**

  Use the yaml config file in the repo to lunch the mediamtx first before going on with the ffmpeg : Use ./mediamtx (Please read the mediamtx documentation for more informations).

  **FFmpeg command for a camera stream simulation (the case if you dont have a camera that can output a video stream for now):**

  ```sh
  ffmpeg -re -stream_loop -1 -i "/path/to/video.dav" -c copy -f rtsp -rtsp_transport tcp rtsp://127.0.0.1:8554/live
  ```

  **FFmpeg command for a camera stream:**

  ```sh
  ffmpeg -rtsp_transport tcp -i "rtsp://username:password@camera_ip:port/stream_path" -c copy -f rtsp -rtsp_transport tcp rtsp://127.0.0.1:8554/live
  ```

  **Lunch the test script:**

  ```sh
  python training_phase/analyse_video_online.py --stream_url "rtsp://127.0.0.1:8554/live" --model_path "path/to/resnet_v2_scratch_vae_best.pth" --use_lpips --threshold 0.14 --frame_sampling_rate 2
  ```
- **Production/Real-time (Linux only):**
  ```sh
  python test_script_linux.py --stream_url "rtsp://127.0.0.1:8554/live" \
    --model_path "/path/to/resnet_v2_scratch_vae_best.pth" \
    --threshold 0.14 \
    --use_lpips \
    --frame_sampling_rate 10 \
    --latent_dim 512 \
    --status_interval 20
  ```

### 4. Web Dashboard (Linux Only)
- **Start the Flask app:**
  ```sh
  cd web_support
  python app.py
  ```
- **Features:**
  - Start/stop live anomaly detection
  - Adjust threshold, frame sampling rate, and status interval on the fly
  - View live logs and anomaly images
  - Browse output files via the built-in file explorer
- **Access:** Open [http://localhost:5000](http://localhost:5000) in your browser.

---

## Useful Commands
See `cmds_usefull.txt` for more advanced usage, streaming with ffmpeg, and MediaMTX setup for RTSP simulation.

**Example: Stream a video as RTSP (for testing):**
```sh
ffmpeg -re -stream_loop -1 -i /path/to/your_video.dav -c copy -f rtsp -rtsp_transport tcp rtsp://127.0.0.1:8554/live
```

---

## Notes on OS Compatibility
- **Linux:**
  - All features supported, including the web dashboard.
  - Recommended for production and real-time use.
  - Use a lightweighted linux version (aka Ubuntu server, arch linux, debian etc ...) try to use a no GUI since this project NEEDS A LOT OF PERFORMANCES, ideally a GPU of 800 cuda cores for a best experience, otherwise a cpu with 2.8Ghz (at least 6 cores) can do the work but you won't be able to surpass 10 images per sec analysis.
- **Windows:**
  - Training, offline, and online analysis scripts are supported.
  - The web dashboard (`web_support/`) is **not** supported on Windows (requires Linux or WSL).
  - Some scripts and environment settings may require manual adjustment.

---

## Contributing
Contributions, issues, and feature requests are welcome! Please open an issue or submit a pull request.

**Contact:** _[bounjoum.noujoum@gmail.com]_

---

## Acknowledgments
- Based on PyTorch, torchvision, and open-source VAE/ResNet implementations, MediaMTX and ffmpeg
- Thanks to the open-source community for tools and inspiration

---

_For more details on each file or to get started with a specific part, see the documentation in each script or open an issue for help._
