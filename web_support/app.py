from flask import Flask, render_template, jsonify, request, send_from_directory, send_file, abort
from test_script_linux7 import analyse_stream
import threading
import queue
import argparse
import builtins
import os

app = Flask(__name__)

# Thread/process and state management
analysis_process = None
stop_event = threading.Event()  
analysis_output = queue.Queue()

# Thread-safe global variables for dynamic parameters
current_threshold = [0.14]              # Anomaly threshold (as a list for mutability)
current_frame_sampling_rate = [10]      # Frame sampling rate (as a list for mutability)
current_status_interval = [5]           # Status interval (as a list for mutability)
threshold_lock = threading.Lock()       # Locks for thread-safe updates
frame_sampling_rate_lock = threading.Lock()
status_interval_lock = threading.Lock()

# Custom print to capture logs from the analysis thread
original_print = builtins.print
def custom_print(*args, **kwargs):
    message = ' '.join(str(arg) for arg in args)
    analysis_output.put(message)        # Push log to queue for GUI
    original_print(*args, **kwargs)     # Still print to console

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_analysis', methods=['POST'])
def start_analysis():
    global analysis_process
    if analysis_process and analysis_process.is_alive():
        return jsonify({'status': 'error', 'message': 'Analysis already running'})

    stop_event.clear()
    builtins.print = custom_print

    # Build dynamic args for the analysis thread
    args = argparse.Namespace(
        stream_url='rtsp://127.0.0.1:8554/live',
        model_path=r'/home/archer/venv/resnet_v2_scratch_vae_best.pth',
        threshold=current_threshold,
        use_lpips=True,
        frame_sampling_rate=current_frame_sampling_rate,
        latent_dim=512,
        input_size=320,
        status_interval=current_status_interval,
        output_normal_dir='./output_normal_online',
        output_anomaly_dir='./output_anomalies_online',
        save_normal_interval_sec=20,
    )

    def run_analysis():
        try:
            analyse_stream(args, stop_event)
        except Exception as e:
            analysis_output.put(f"Error: {str(e)}")
        finally:
            builtins.print = original_print

    analysis_process = threading.Thread(target=run_analysis)
    analysis_process.daemon = True
    analysis_process.start()
    return jsonify({'status': 'success', 'message': 'Analysis started'})

# Endpoint to update threshold dynamically
@app.route('/update_threshold', methods=['POST'])
def update_threshold():
    data = request.get_json()
    if not data or 'threshold' not in data:
        return jsonify({'status': 'error', 'message': 'Aucun seuil fourni'})
    try:
        new_threshold = float(data['threshold'])
        if not 0 <= new_threshold <= 1:
            return jsonify({'status': 'error', 'message': 'Veuillez entrer un nombre entre 0 et 1'})
        with threshold_lock:
            current_threshold[0] = new_threshold
            analysis_output.put(f"Seuil mis à jour à {new_threshold:.2f}")
        return jsonify({'status': 'success', 'message': f'Seuil mis à jour à {new_threshold:.2f}'})
    except ValueError:
        return jsonify({'status': 'error', 'message': 'Valeur de seuil invalide'})

# Endpoint to update frame_sampling_rate dynamically
@app.route('/update_frame_sampling_rate', methods=['POST'])
def update_frame_sampling_rate():
    data = request.get_json()
    if not data or 'frame_sampling_rate' not in data:
        return jsonify({'status': 'error', 'message': 'Aucun frame sampling rate fourni'})
    try:
        new_rate = int(data['frame_sampling_rate'])
        if new_rate not in [10, 15, 20, 25, 30]:
            return jsonify({'status': 'error', 'message': 'Frame sampling rate doit être 10, 15, 20, 25 ou 30'})
        with frame_sampling_rate_lock:
            current_frame_sampling_rate[0] = new_rate
            analysis_output.put(f"Frame sampling rate mis à jour à {new_rate}")
        return jsonify({'status': 'success', 'message': f'Frame sampling rate mis à jour à {new_rate}'})
    except ValueError:
        return jsonify({'status': 'error', 'message': 'Valeur de frame sampling rate invalide'})

# Endpoint to update status_interval dynamically
@app.route('/update_status_interval', methods=['POST'])
def update_status_interval():
    data = request.get_json()
    if not data or 'status_interval' not in data:
        return jsonify({'status': 'error', 'message': 'Aucun status interval fourni'})
    try:
        new_interval = int(data['status_interval'])
        if new_interval not in [5, 10, 15, 20, 25, 30]:
            return jsonify({'status': 'error', 'message': 'Status interval doit être 5, 10, 15, 20, 25 ou 30'})
        with status_interval_lock:
            current_status_interval[0] = new_interval
            analysis_output.put(f"Status interval mis à jour à {new_interval}")
        return jsonify({'status': 'success', 'message': f'Status interval mis à jour à {new_interval}'})
    except ValueError:
        return jsonify({'status': 'error', 'message': 'Valeur de status interval invalide'})

@app.route('/stop_analysis', methods=['POST'])
def stop_analysis():
    global analysis_process
    if analysis_process and analysis_process.is_alive():
        stop_event.set()
        analysis_process.join(timeout=5)
        builtins.print = original_print
        analysis_output.put("Analysis stopped.")
        return jsonify({'status': 'success', 'message': 'Analysis stopped'})
    return jsonify({'status': 'success', 'message': 'No analysis running'})

# Endpoint to get logs for the GUI
@app.route('/status')
def status():
    logs = []
    while not analysis_output.empty():
        logs.append(analysis_output.get())
    return jsonify({'logs': logs})

# File browser and image serving
@app.route('/filesystem')
def filesystem():
    root_path = r'/home/archer/venv_web_v'
    allowed_dirs = ['output_normal_online', 'output_anomalies_online']
    req_path = request.args.get('path', root_path)
    abs_req_path = os.path.abspath(req_path)
    parent_path = None
    if abs_req_path != os.path.abspath(root_path):
        parent_path = os.path.dirname(abs_req_path)
    if abs_req_path == os.path.abspath(root_path):
        items = []
        for d in allowed_dirs:
            dir_path = os.path.join(root_path, d)
            if os.path.isdir(dir_path):
                items.append({'name': d, 'path': dir_path, 'is_dir': True})
        items.sort(key=lambda x: x['name'])
        return render_template('filesystem.html', current_path=root_path, items=items, root_path=root_path, parent_path=parent_path)
    elif any(abs_req_path.startswith(os.path.join(root_path, d)) for d in allowed_dirs):
        try:
            items = []
            for item in os.listdir(abs_req_path):
                item_path = os.path.join(abs_req_path, item)
                if os.path.isdir(item_path):
                    items.append({'name': item, 'path': item_path, 'is_dir': True})
                elif os.path.isfile(item_path) and item.lower().endswith('.png'):
                    items.append({'name': item, 'path': item_path, 'is_dir': False})
            items.sort(key=lambda x: (not x['is_dir'], x['name']))
            return render_template('filesystem.html', current_path=abs_req_path, items=items, root_path=root_path, parent_path=parent_path)
        except Exception as e:
            return render_template('filesystem.html', current_path=abs_req_path, items=[], root_path=root_path, parent_path=parent_path, error=str(e))
    else:
        return render_template('filesystem.html', current_path=root_path, items=[], root_path=root_path, parent_path=parent_path, error='Access denied.')

@app.route('/images/<path:filepath>')
def images(filepath):
    # Only allow certain base directories for security
    allowed_dirs = [
        '/home/archer/venv_web_v/output_anomalies_online',
        '/home/archer/venv_web_v/output_normal_online'
    ]
    # Reconstruct the absolute path
    abs_path = '/' + filepath if not filepath.startswith('/') else filepath

    # Check if the path is within allowed directories
    if any(abs_path.startswith(d) for d in allowed_dirs):
        try:
            return send_file(abs_path)
        except Exception as e:
            print(e)
            abort(404)
    else:
        abort(403)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
