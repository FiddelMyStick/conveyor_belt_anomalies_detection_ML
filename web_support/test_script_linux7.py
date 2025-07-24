
# --- Standard and ML/vision imports ---
import os
import time
import argparse
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2
import threading
import queue

# --- Model import (ResNetVAE_V2) ---
try:
    from models.resnet import ResNetVAE_V2
except ImportError:
    print("ERROR: Could not import ResNetVAE_V2 from model.py.")
    exit(1)

# --- Optional LPIPS import for perceptual loss ---
try:
    import lpips
    lpips_available = True
except ImportError:
    lpips_available = False

# --- Utility: Create a composite image for anomaly visualization ---
def create_comparison_image(original_img, recon_img, score, score_name="MAE"):
    # Convert tensors to numpy images, denormalize
    original_img = original_img.squeeze().cpu().permute(1, 2, 0).numpy() * 0.5 + 0.5
    recon_img = recon_img.squeeze().cpu().permute(1, 2, 0).numpy() * 0.5 + 0.5
    original_img = np.clip(original_img, 0, 1)
    recon_img = np.clip(recon_img, 0, 1)

    # Compute difference heatmap
    diff = np.abs(original_img - recon_img)
    heatmap_gray = np.mean(diff, axis=2)
    heatmap_uint8 = np.uint8(255 * heatmap_gray)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color_rgb = heatmap_color[..., ::-1]

    # Plot original, reconstruction, and heatmap
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Anomaly Detected - {score_name}: {score:.4f}", fontsize=14)
    axs[0].imshow(original_img); axs[0].set_title('Original'); axs[0].axis('off')
    axs[1].imshow(recon_img); axs[1].set_title('Reconstruction'); axs[1].axis('off')
    axs[2].imshow(heatmap_color_rgb); axs[2].set_title(f'Difference Heatmap'); axs[2].axis('off')
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Convert plot to numpy image
    try:
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        img_plot_rgba = np.asarray(buf)
        img_plot_rgb = img_plot_rgba[..., :3]
    finally:
        plt.close(fig)
    return img_plot_rgb

# --- Threaded video stream reader for non-blocking frame capture ---
class VideoStreamReader:
    def __init__(self, src=0, max_queue_size=64):
        self.stream = cv2.VideoCapture(src)
        if not self.stream.isOpened():
            raise ValueError(f"Could not open video stream at {src}")
        self.q = queue.LifoQueue(maxsize=max_queue_size)
        self.stopped = False
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True

    def start(self):
        self.thread.start()
        return self

    def update(self):
        while not self.stopped:
            ret, frame = self.stream.read()
            if not ret:
                self.stop()
                return
            if not self.q.full():
                self.q.put(frame)
            else:
                try: self.q.get_nowait()
                except queue.Empty: pass
                self.q.put(frame)

    def read(self):
        return self.q.get()

    def stop(self):
        self.stopped = True
        if self.thread.is_alive():
            self.thread.join()
        self.stream.release()

# --- Main analysis function: runs the anomaly detection loop ---
def analyse_stream(args, stop_event=None):
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(device)}")
    else:
        print("CUDA not available. Using CPU.")

    # Ensure output directories exist
    os.makedirs(args.output_normal_dir, exist_ok=True)
    os.makedirs(args.output_anomaly_dir, exist_ok=True)

    # Image preprocessing pipeline
    transform = transforms.Compose([
        transforms.Resize((args.input_size, args.input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Load model
    print(f"Loading model: {args.model_path}")
    model = ResNetVAE_V2(latent_dim=args.latent_dim, input_height=args.input_size)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()

    # Optionally load LPIPS
    lpips_fn = None
    if args.use_lpips:
        if not lpips_available:
            print("WARNING: --use_lpips flag set but lpips library not found. Using MAE instead.")
            args.use_lpips = False
        else:
            try:
                print("Loading LPIPS model...")
                lpips_fn = lpips.LPIPS(net='alex', verbose=False).to(device)
                for param in lpips_fn.parameters(): param.requires_grad = False
                lpips_fn.eval()
            except Exception as e:
                print(f"Could not load LPIPS model: {e}. Falling back to MAE.")
                args.use_lpips = False

    # Print current parameters
    score_name = "LPIPS" if args.use_lpips else "MAE"
    print(f"Using {score_name} score for thresholding.")
    print(f"Anomaly Threshold ({score_name}): {args.threshold[0]}")
    print(f"Frame Sampling Rate: {args.frame_sampling_rate[0]}")
    print(f"Status Interval: {args.status_interval[0]}")

    print(f"Connecting to video stream: {args.stream_url}")
    try:
        stream_reader = VideoStreamReader(src=args.stream_url).start()
        print("Stream connection successful. Starting analysis...")
        time.sleep(2.0)
    except ValueError as e:
        print(f"ERROR: {e}")
        return

    # Main loop variables
    source_frame_count = 0
    analyzed_frame_count = 0
    anomalies_detected = 0
    last_normal_save_time = time.time()
    processing_times = []

    print("\nStarting continuous analysis... Press Ctrl+C to stop if you are on cmd line, else stop from the GUI.")
    try:
        while True:
            # Check for stop event (from GUI)
            if stop_event is not None and stop_event.is_set():
                print("Stop event detected. Exiting analysis loop.")
                break
            frame = stream_reader.read()
            if frame is None: print("Stream ended."); break
            source_frame_count += 1

            # Only process every Nth frame
            if source_frame_count % args.frame_sampling_rate[0] != 0: continue
            
            analyzed_frame_count += 1
            start_time = time.time()
            
            # Preprocess frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            input_tensor = transform(frame_pil).unsqueeze(0).to(device)

            # Forward pass
            with torch.no_grad():
                recons_tensor, _, _, _ = model(input_tensor)
                if args.use_lpips and lpips_fn:
                    score_to_use = lpips_fn(recons_tensor, input_tensor).item()
                else:
                    score_to_use = F.l1_loss(recons_tensor, input_tensor).item()

            # Anomaly detection logic
            if score_to_use >= args.threshold[0]:
                anomalies_detected += 1
                print(f"-> ANOMALY DETECTED! Frame Source: {source_frame_count}, {score_name}: {score_to_use:.4f} (Threshold: {args.threshold[0]:.2f})")
                
                vis_image = create_comparison_image(input_tensor, recons_tensor, score_to_use, score_name)
                save_filename = f"anomaly_frame_{source_frame_count:08d}_{score_name}_{score_to_use:.4f}.png"
                save_path = os.path.join(args.output_anomaly_dir, save_filename)
                cv2.imwrite(save_path, vis_image[..., ::-1])
            else:
                current_time = time.time()
                if current_time - last_normal_save_time >= args.save_normal_interval_sec:
                    save_filename = f"normal_frame_{source_frame_count:08d}.png"
                    save_path = os.path.join(args.output_normal_dir, save_filename)
                    cv2.imwrite(save_path, frame)
                    last_normal_save_time = current_time

            end_time = time.time()
            processing_times.append(end_time - start_time)

            # Print status every status_interval frames
            if analyzed_frame_count % args.status_interval[0] == 0 and analyzed_frame_count > 0:
                rolling_avg_time = np.mean(processing_times[-args.status_interval[0]:])
                rolling_fps = 1.0 / rolling_avg_time if rolling_avg_time > 0 else 0
                print(f"--- STATUS | Analyzed Frames: {analyzed_frame_count} | Total Anomalies: {anomalies_detected} | Current FPS: {rolling_fps:.1f} ---")

    except KeyboardInterrupt:
        print("\nCtrl+C detected. Shutting down gracefully...")
    finally:
        print("\n--- Final Analysis Summary ---")
        if processing_times:
             avg_proc_time = np.mean(processing_times)
             fps_proc = 1.0 / avg_proc_time if avg_proc_time > 0 else 0
             print(f"Average processing time per analyzed frame: {avg_proc_time*1000:.2f} ms (~{fps_proc:.1f} FPS)")
        print(f"Total anomalies detected: {anomalies_detected}")
        print("Stopping video stream reader...")
        if 'stream_reader' in locals():
            stream_reader.stop()

# --- CLI entrypoint for standalone use ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Online Video Anomaly Detection using a VAE")
    parser.add_argument('--stream_url', type=str, required=True, help='RTSP URL of the live video stream')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained VAE model .pth file')
    parser.add_argument('--output_normal_dir', type=str, default='./output_normal_online', help='Directory for normal frames')
    parser.add_argument('--output_anomaly_dir', type=str, default='./output_anomalies_online', help='Directory for anomaly visualizations')
    parser.add_argument('--latent_dim', type=int, default=512, help='Latent dimension of the VAE model')
    parser.add_argument('--input_size', type=int, default=320, help='Input image size')
    parser.add_argument('--threshold', type=float, required=True, help='Anomaly detection threshold. NEEDS CALIBRATION!')
    parser.add_argument('--frame_sampling_rate', type=int, default=1, metavar='N', help='Process only every Nth frame (default: 1)')
    parser.add_argument('--save_normal_interval_sec', type=float, default=5.0, metavar='S', help='Interval in seconds to save a normal frame (default: 5.0)')
    parser.add_argument('--use_lpips', action='store_true', help='Use LPIPS score for thresholding instead of MAE')
    parser.add_argument('--status_interval', type=int, default=50, metavar='F', help='Print a status update every F analyzed frames (default: 50)')

    args = parser.parse_args()
    analyse_stream(args)