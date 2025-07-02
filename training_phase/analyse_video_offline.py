# analyse_video.py

import os
import time
import argparse # To handle command-line arguments
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from PIL import Image
import cv2 # OpenCV for video reading and heatmap coloring
from tqdm import tqdm # Progress bar for video frames

# --- Import the VAE Model --- #
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

# --- Main Analysis Function --- #

def analyse_video(args):
    """Performs anomaly detection on a video using a VAE model."""

    # --- Setup Device ---
    if args.force_cpu:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- Create Output Directories ---
    try:
        os.makedirs(args.output_normal_dir, exist_ok=True)
        os.makedirs(args.output_anomaly_dir, exist_ok=True)
        print(f"Normal frames output: {args.output_normal_dir}")
        print(f"Anomaly frames output: {args.output_anomaly_dir}")
    except OSError as e:
        print(f"ERROR creating output directories: {e}")
        return

    # --- Prepare Transforms ---
    transform = transforms.Compose([
        transforms.Resize((args.input_size, args.input_size)),
        transforms.ToTensor(), # To [0, 1] range
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # To [-1, 1] range
    ])

    # --- Load Model ---
    print(f"Loading model definition (latent_dim={args.latent_dim})...")
    # Instantiate the specific model architecture class used for training
    model = ResNetVAE_V2(latent_dim=args.latent_dim, input_height=args.input_size)

    if not os.path.exists(args.model_path):
        print(f"ERROR: Model checkpoint not found at {args.model_path}")
        return

    print(f"Loading weights from: {args.model_path}")
    try:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model.to(device)
        model.eval() # Set to evaluation mode! Important!
        print("Model loaded successfully!")
    except Exception as e:
        print(f"ERROR loading model state_dict: {e}")
        print("Ensure the model definition/latent_dim matches the checkpoint.")
        return

    # --- Optional: Load LPIPS ---
    lpips_fn = None
    if args.use_lpips and lpips_available:
        try:
            print("Loading LPIPS model...")
            # Using net='alex' is generally faster than 'vgg'
            lpips_fn = lpips.LPIPS(net='alex', verbose=False).to(device)
            for param in lpips_fn.parameters(): param.requires_grad = False
            lpips_fn.eval() # Set LPIPS to eval mode too
            print("LPIPS model loaded.")
            print("Using LPIPS score for thresholding.")
        except Exception as e:
            print(f"Could not load LPIPS model: {e}. Falling back to MAE.")
            lpips_fn = None; args.use_lpips = False
    if not args.use_lpips:
        print("Using Mean Absolute Error (MAE) for thresholding.")


    # --- Open Video ---
    if not os.path.exists(args.video_path):
        print(f"ERROR: Video file not found: {args.video_path}")
        return

    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        print(f"ERROR: Could not open video file: {args.video_path}")
        return

    # --- Video Properties ---
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        print("WARNING: Could not determine video FPS. Using frame interval for normal save.")
        save_normal_frame_interval = 30 # Default frame interval if FPS unknown
        interval_unit = "frames"
    else:
        save_normal_frame_interval = max(1, int(round(args.save_normal_interval_sec * fps)))
        interval_unit = f"seconds (~{save_normal_frame_interval} frames)"
    print(f"Video Info: Total Frames ~{total_frames}, FPS ~{fps if fps else 'N/A'}")
    print(f"Processing every {args.frame_sampling_rate} frames.")
    print(f"Saving one normal frame approx. every {args.save_normal_interval_sec} {interval_unit}.")
    print(f"Anomaly Threshold ({'LPIPS' if args.use_lpips else 'MAE'}): {args.threshold}")


    # --- Processing Loop ---
    frame_number = 0
    saved_frame_count = 0
    last_normal_save_frame = -save_normal_frame_interval # Ensure first normal frame can be saved
    anomalies_detected = 0
    processing_times = []

    # Determine number of frames to process based on sampling
    effective_total_frames = (total_frames // args.frame_sampling_rate) if total_frames > 0 else 0

    pbar = tqdm(total=effective_total_frames, desc="Analyzing Video", unit="frame")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_number += 1

        # --- Frame Sampling ---
        if (frame_number - 1) % args.frame_sampling_rate != 0: # Adjust logic slightly to process frame 1 if rate is > 1
            continue

        pbar.update(1) # Update progress bar only for processed frames

        try:
            start_time = time.time()

            # --- Preprocessing ---
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            input_tensor = transform(frame_pil).unsqueeze(0).to(device)

            # --- Inference ---
            with torch.no_grad():
                recons_tensor, _, _, _ = model(input_tensor)

            # --- Calculate Error Metrics ---
            # Ensure both tensors are in [-1, 1] for calculation
            mae_error = F.l1_loss(recons_tensor, input_tensor).item()

            lpips_score = -1.0
            if lpips_fn is not None:
                with torch.no_grad(): # Ensure no grads for LPIPS calculation
                    lpips_score = lpips_fn(recons_tensor, input_tensor).item()

            end_time = time.time()
            processing_times.append(end_time - start_time)

            # --- Thresholding & Saving ---
            score_to_use = lpips_score if args.use_lpips else mae_error
            score_name = "LPIPS" if args.use_lpips else "MAE"

            if score_to_use >= args.threshold:
                # Anomaly Detected
                anomalies_detected += 1
                vis_image = create_comparison_image(input_tensor, recons_tensor, score_to_use, score_name)
                save_filename = f"anomaly_frame_{frame_number:06d}_{score_name}_{score_to_use:.4f}.png"
                save_path = os.path.join(args.output_anomaly_dir, save_filename)
                # Convert RGB plot image to BGR for cv2.imwrite
                cv2.imwrite(save_path, vis_image[..., ::-1])

            else:
                # Normal Frame - Save periodically
                if frame_number >= last_normal_save_frame + save_normal_frame_interval:
                    save_filename = f"normal_frame_{frame_number:06d}.png"
                    save_path = os.path.join(args.output_normal_dir, save_filename)
                    cv2.imwrite(save_path, frame) # Save original frame (BGR)
                    last_normal_save_frame = frame_number
                    saved_frame_count += 1

        except Exception as e:
            print(f"\nERROR processing frame {frame_number}: {e}")
            # You might want to add specific error handling or logging here

    # --- End of Loop ---
    pbar.close()
    cap.release()
    print("\n--- Video Processing Finished ---")
    print(f"Total frames read: {frame_number}")
    print(f"Total frames analyzed (sampled): {len(processing_times)}")
    print(f"Anomalies detected (Threshold: {args.threshold} on {score_name}): {anomalies_detected}")
    print(f"Normal frames saved (approx every {args.save_normal_interval_sec}s): {saved_frame_count}")
    if processing_times:
        avg_proc_time = np.mean(processing_times)
        fps_proc = 1.0 / avg_proc_time if avg_proc_time > 0 else 0
        print(f"Average processing time per analyzed frame: {avg_proc_time*1000:.2f} ms (~{fps_proc:.1f} FPS)")
    print(f"Normal frames saved in: {args.output_normal_dir}")
    print(f"Anomaly visualizations saved in: {args.output_anomaly_dir}")


# --- Argument Parsing and Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video Anomaly Detection using ResNetVAE_V2")

    parser.add_argument('--video_path', type=str, required=True, help='Path to the input video file (MP4, AVI, etc.)')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained VAE model .pth state_dict file')
    parser.add_argument('--output_normal_dir', type=str, default='./output_normal', help='Directory to save normal frames')
    parser.add_argument('--output_anomaly_dir', type=str, default='./output_anomalies', help='Directory to save anomaly visualizations')
    parser.add_argument('--latent_dim', type=int, default=512, help='Latent dimension of the loaded VAE model')
    parser.add_argument('--input_size', type=int, default=320, help='Input image size the model expects (e.g., 320)')
    parser.add_argument('--threshold', type=float, required=True, help='Anomaly detection threshold (MAE or LPIPS score). NEEDS CALIBRATION!')
    parser.add_argument('--frame_sampling_rate', type=int, default=1, metavar='N', help='Process only every Nth frame (1 = process all) (default: 1)')
    parser.add_argument('--save_normal_interval_sec', type=float, default=1.0, metavar='S', help='Approx. interval in seconds to save a normal frame (default: 1.0)')
    parser.add_argument('--use_lpips', action='store_true', help='Use LPIPS score for thresholding instead of MAE (requires lpips library)')
    parser.add_argument('--force_cpu', action='store_true', help='Force CPU usage even if CUDA is available')

    args = parser.parse_args()

    # Basic validation
    if not os.path.exists(args.video_path): parser.error(f"Video file not found: {args.video_path}")
    if not os.path.exists(args.model_path): parser.error(f"Model file not found: {args.model_path}")
    if args.frame_sampling_rate < 1: parser.error("--frame_sampling_rate must be >= 1")
    if args.save_normal_interval_sec <= 0: parser.error("--save_normal_interval_sec must be > 0")
    if args.use_lpips and not lpips_available: print("WARNING: --use_lpips flag set but lpips library not found. Using MAE instead."); args.use_lpips = False

    # Run the analysis
    analyse_video(args)