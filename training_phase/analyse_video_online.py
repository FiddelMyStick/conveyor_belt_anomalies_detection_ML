# analyse_video_online.py

import os
import time
import argparse
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm
import threading
import queue

#--- Import the VAE Model --- #
# Assumes model.py is in the same directory or Python path
try:
    from models.resnet import ResNetVAE_V2
except ImportError:
    print("ERROR: Could not import ResNetVAE_V2 from model.py.")
    print("Ensure model.py is in the same directory or accessible in PYTHONPATH.")
    exit(1)

# --- Optional: LPIPS --- #
try:
    import lpips
    lpips_available = True
except ImportError:
    # print("INFO: lpips library not found. Perceptual score calculation disabled.")
    # print("Install using: pip install lpips")
    lpips_available = False

# +++ NEW: Frame Grabber Thread +++
class VideoStreamReader:
    """
    A threaded class to read frames from a cv2.VideoCapture source
    and keep a buffer of the most recent frames in a queue.
    """
    def __init__(self, src=0, max_queue_size=128):
        self.stream = cv2.VideoCapture(src)
        if not self.stream.isOpened():
            print(f"ERROR: Could not open video stream at {src}")
            raise ValueError("Could not open video stream")

        # Use a LIFO queue to always get the most recent frame, discarding old ones if analysis is slow
        self.q = queue.LifoQueue(maxsize=max_queue_size)
        self.stopped = False
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True # Thread dies when main program exits

    def start(self):
        self.thread.start()
        return self

    def update(self):
        while not self.stopped:
            ret, frame = self.stream.read()
            if not ret:
                print("Stream ended or disconnected. Stopping thread.")
                self.stop()
                return

            if not self.q.full():
                self.q.put(frame) # Add frame to queue
            else:
                # If queue is full, pop the oldest and add the newest
                try:
                    self.q.get_nowait()
                except queue.Empty:
                    pass
                self.q.put(frame)

    def read(self):
        # Return the most recent frame from the queue
        return self.q.get()

    def stop(self):
        self.stopped = True
        self.thread.join() # Wait for thread to finish
        self.stream.release()


# --- Helper Function for Visualization --- #

def create_comparison_image(original_img, recon_img, score, score_name="MAE"):
    """Creates a comparison image: Original | Reconstructed | Heatmap"""
    # Ensure inputs are numpy arrays in [0, 1] range, HWC format
    if isinstance(original_img, torch.Tensor):
        # De-normalize [-1, 1] -> [0, 1] and convert to HWC numpy
        original_img = original_img.squeeze().cpu().permute(1, 2, 0).numpy() * 0.5 + 0.5
    if isinstance(recon_img, torch.Tensor):
        # De-normalize [-1, 1] -> [0, 1] and convert to HWC numpy
        recon_img = recon_img.squeeze().cpu().permute(1, 2, 0).numpy() * 0.5 + 0.5

    original_img = np.clip(original_img, 0, 1)
    recon_img = np.clip(recon_img, 0, 1)

    # Calculate difference and heatmap
    diff = np.abs(original_img - recon_img)
    heatmap_gray = np.mean(diff, axis=2) # Mean difference across channels
    # Normalize heatmap for consistent color mapping across frames (optional but good)
    # Or use vmin=0, vmax=1 for imshow norm if differences are expected in [0,1]
    norm_heatmap = mcolors.Normalize(vmin=0.0, vmax=max(0.1, heatmap_gray.max())) # Avoid empty range, cap minimum max at 0.1
    heatmap_uint8 = np.uint8(plt.cm.viridis(norm_heatmap(heatmap_gray))[..., :3] * 255) # Use viridis colormap
    # Convert colormap result from RGB to BGR for potential OpenCV use later if needed
    # For direct matplotlib display, RGB is fine. Let's keep RGB.

    # Create plot
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Anomaly Detected - {score_name}: {score:.4f}", fontsize=14)

    axs[0].imshow(original_img)
    axs[0].set_title('Original')
    axs[0].axis('off')

    axs[1].imshow(recon_img)
    axs[1].set_title('Reconstruction')
    axs[1].axis('off')

    im = axs[2].imshow(heatmap_gray, cmap='viridis', norm=norm_heatmap) # Show grayscale heatmap with colormap
    axs[2].set_title(f'Difference Heatmap')
    axs[2].axis('off')
    fig.colorbar(im, ax=axs[2], fraction=0.046, pad=0.04) # Add colorbar

    plt.tight_layout(rect=[0, 0, 1, 0.95]) # Adjust layout for title

    # Convert plot to numpy array (RGB)
    try:
        fig.canvas.draw() # Ensure figure is rendered
        # Get RGBA buffer
        buf = fig.canvas.buffer_rgba()
        # Convert buffer to numpy array
        img_plot_rgba = np.asarray(buf)
        # Keep only RGB channels
        img_plot_rgb = img_plot_rgba[..., :3]
    finally:
        # *** CRUCIAL: Ensure plot is closed even if buffer access fails ***
        plt.close(fig)

    return img_plot_rgb # Return the RGB numpy array

# (The create_comparison_image function should be here)
# ...

def analyse_stream(args):
    """Performs real-time anomaly detection on a video stream."""

    # --- Setup Device ---
    device = torch.device('cuda' if torch.cuda.is_available() and not args.force_cpu else 'cpu')
    print(f"Using device: {device}")

    # --- Create Output Directories ---
    os.makedirs(args.output_normal_dir, exist_ok=True)
    os.makedirs(args.output_anomaly_dir, exist_ok=True)

    # --- Prepare Transforms ---
    transform = transforms.Compose([
        transforms.Resize((args.input_size, args.input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # --- Load Model ---
    print(f"Loading model: {args.model_path}")
    model = ResNetVAE_V2(latent_dim=args.latent_dim, input_height=args.input_size)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()

    # --- Start Video Stream Reader ---
    print(f"Connecting to video stream: {args.stream_url}")
    try:
        stream_reader = VideoStreamReader(src=args.stream_url).start()
        print("Stream connection successful. Starting analysis...")
        time.sleep(2.0) # Give stream time to buffer
    except ValueError:
        return # Exit if stream connection failed

    # --- Main Analysis Loop ---
    frame_count = 0
    anomalies_detected = 0
    last_normal_save_time = time.time()

    try:
        while True: # Loop forever
            # Get the most recent frame from the grabber thread
            frame = stream_reader.read()

            if frame is None:
                print("Received empty frame, maybe stream ended?")
                break

            frame_count += 1

            # --- Frame Sampling ---
            if frame_count % args.frame_sampling_rate != 0:
                continue

            # --- Preprocessing ---
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            input_tensor = transform(frame_pil).unsqueeze(0).to(device)

            # --- Inference & Error Calculation ---
            with torch.no_grad():
                recons_tensor, _, _, _ = model(input_tensor)
                mae_error = F.l1_loss(recons_tensor, input_tensor).item()

            # --- Thresholding & Saving ---
            if mae_error >= args.threshold:
                # Anomaly Detected
                anomalies_detected += 1
                vis_image = create_comparison_image(input_tensor, recons_tensor, mae_error, "MAE")
                save_filename = f"anomaly_frame_{frame_count:08d}_MAE_{mae_error:.4f}.png"
                save_path = os.path.join(args.output_anomaly_dir, save_filename)
                cv2.imwrite(save_path, vis_image[..., ::-1])
                print(f"Anomaly detected! Frame {frame_count}, MAE: {mae_error:.4f}. Saved to {save_path}")

            else:
                # Normal Frame - Save periodically
                current_time = time.time()
                if current_time - last_normal_save_time >= args.save_normal_interval_sec:
                    save_filename = f"normal_frame_{frame_count:08d}.png"
                    save_path = os.path.join(args.output_normal_dir, save_filename)
                    cv2.imwrite(save_path, frame)
                    last_normal_save_time = current_time

    except KeyboardInterrupt:
        print("\nCtrl+C detected. Shutting down...")
    finally:
        # --- Cleanup ---
        print("Stopping video stream reader...")
        stream_reader.stop()
        print("Analysis finished.")
        print(f"Total frames processed (sampled): {frame_count // args.frame_sampling_rate}")
        print(f"Total anomalies detected: {anomalies_detected}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Online Video Anomaly Detection using a VAE")
    
    # Redefined arguments for online context
    parser.add_argument('--stream_url', type=str, required=True, help='RTSP URL of the live video stream')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained VAE model .pth file')
    parser.add_argument('--output_normal_dir', type=str, default='./output_normal_online', help='Directory to save normal frames')
    parser.add_argument('--output_anomaly_dir', type=str, default='./output_anomalies_online', help='Directory to save anomaly visualizations')
    parser.add_argument('--latent_dim', type=int, default=512, help='Latent dimension of the VAE model')
    parser.add_argument('--input_size', type=int, default=320, help='Input image size')
    parser.add_argument('--threshold', type=float, required=True, help='MAE anomaly detection threshold. NEEDS CALIBRATION!')
    parser.add_argument('--frame_sampling_rate', type=int, default=1, metavar='N', help='Process only every Nth frame (default: 1)')
    parser.add_argument('--save_normal_interval_sec', type=float, default=5.0, metavar='S', help='Interval in seconds to save a normal frame (default: 5.0)')
    parser.add_argument('--force_cpu', action='store_true', help='Force CPU usage')

    args = parser.parse_args()
    
    analyse_stream(args)