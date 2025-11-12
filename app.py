import numpy as np
import math
from scipy.io import wavfile
import matplotlib.pyplot as plt
import os
import warnings
import io
import base64
from flask import Flask, request, jsonify, render_template, send_file
from werkzeug.utils import secure_filename
import tempfile

# Suppress DeprecationWarning for scipy.io.wavfile.read
warnings.filterwarnings("ignore", category=DeprecationWarning)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()

# ----------------------------
# Audio Processing Functions
# ----------------------------

def to_float32(x):
    orig_dtype = x.dtype
    if np.issubdtype(orig_dtype, np.floating):
        y = np.clip(x.astype(np.float32), -1.0, 1.0)
        return y, orig_dtype, None
    if orig_dtype == np.int16:
        y = x.astype(np.float32) / 32768.0
        return y, orig_dtype, 32768.0
    if orig_dtype == np.int32:
        y = x.astype(np.float32) / 2147483648.0
        return y, orig_dtype, 2147483648.0
    if orig_dtype == np.uint8:
        y = (x.astype(np.float32) - 128.0) / 128.0
        return y, orig_dtype, 128.0
    y = x.astype(np.float32)
    y /= np.max(np.abs(y)) + 1e-12
    return y, orig_dtype, None

def from_float32(y, target_dtype, scale_info):
    y = np.clip(y, -1.0, 1.0)
    if np.issubdtype(target_dtype, np.floating):
        return y.astype(target_dtype)
    if target_dtype == np.int16:
        return (y * 32767.0).astype(np.int16)
    if target_dtype == np.int32:
        return (y * 2147483647.0).astype(np.int32)
    if target_dtype == np.uint8:
        return np.clip((y * 128.0) + 128.0, 0, 255).astype(np.uint8)
    return (y * 32767.0).astype(np.int16)

def process_audio_online(y, order=4, lam=0.995, eta=3.0, beta=0.99, delta=1000.0, max_run=4):
    n_samples = len(y)
    r = order
    
    cleaned_audio = np.copy(y)
    a = np.zeros(r, dtype=np.float64)
    P = np.eye(r, dtype=np.float64) * delta
    error_var_estimate = 0.01
    idx_vec = np.arange(1, r + 1)
    
    a_hist = np.zeros((n_samples, r), dtype=np.float64)
    err = np.zeros(n_samples, dtype=np.float64)
    thresholds = np.zeros(n_samples, dtype=np.float64)
    bad_mask = np.zeros(n_samples, dtype=bool)

    k = r
    while k < n_samples - max_run:
        y_past = cleaned_audio[k - idx_vec]
        y_pred = float(np.dot(a, y_past))
        error = y[k] - y_pred

        error_var_estimate = beta * error_var_estimate + (1 - beta) * error**2
        delta_e = math.sqrt(max(error_var_estimate, 1e-12))
        threshold = eta * delta_e
        
        err[k] = error
        thresholds[k] = threshold
        
        if abs(error) > threshold:
            bad_mask[k] = True
            click_start = k
            click_end = k
            
            for j in range(1, max_run):
                if k + j >= n_samples - 1:
                    break
                
                next_y_past = cleaned_audio[k + j - idx_vec]
                next_y_pred = np.dot(a, next_y_past)
                next_error = y[k + j] - next_y_pred
                
                if abs(next_error) > threshold:
                    click_end = k + j
                    bad_mask[k + j] = True
                else:
                    break
            
            num_bad_samples = click_end - click_start + 1
            y_start_good = cleaned_audio[click_start - 1]
            y_end_good = y[click_end + 1]
            
            for i in range(num_bad_samples):
                idx_to_replace = click_start + i
                alpha = (i + 1) / (num_bad_samples + 1)
                cleaned_audio[idx_to_replace] = (1 - alpha) * y_start_good + alpha * y_end_good

            last_valid_a = a_hist[k-1] if k > r else a
            for i in range(click_start, click_end + 1):
                a_hist[i] = last_valid_a
            
            k = click_end + 1
            continue

        Px = P @ y_past
        g_denom = lam + float(y_past.T @ Px)
        K = Px / (g_denom + 1e-12)

        a = a + K * error
        P = (P - np.outer(K, y_past) @ P) / lam
        
        a_hist[k] = a
        k += 1

    last_good_a = a
    for k_fill in range(k, n_samples):
        a_hist[k_fill] = last_good_a
        
    first_valid_a = a_hist[r]
    for k_fill in range(r):
        a_hist[k_fill] = first_valid_a

    return cleaned_audio, a_hist, err, thresholds, bad_mask

def generate_plot_base64(y_orig, y_clean, sr, bad_mask, err, thr):
    """Generate plot and return as base64 string"""
    plt.style.use('seaborn-v0_8-darkgrid')
    n_samples = len(y_orig)
    time_axis = np.arange(n_samples) / sr
    
    detected_clicks_indices = np.where(bad_mask)[0]
    if detected_clicks_indices.size > 0:
        center_of_plot = detected_clicks_indices[len(detected_clicks_indices)//2]
        plot_range_start = max(0, center_of_plot - 1000)
        plot_range_end = min(n_samples, center_of_plot + 1000)
    else:
        plot_range_start = max(0, n_samples // 2 - 1000)
        plot_range_end = min(n_samples, n_samples // 2 + 1000)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Waveform comparison
    ax1.plot(time_axis[plot_range_start:plot_range_end], y_orig[plot_range_start:plot_range_end],
             label='Original Signal', color='red', alpha=0.7, linewidth=1)
    ax1.plot(time_axis[plot_range_start:plot_range_end], y_clean[plot_range_start:plot_range_end],
             label='Cleaned Signal', color='blue', alpha=0.7, linewidth=1)
    ax1.set_title('Waveform Comparison (Click Region)')
    ax1.set_ylabel('Amplitude')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Prediction error with thresholds
    ax2.plot(time_axis[plot_range_start:plot_range_end], err[plot_range_start:plot_range_end], 
             label='Prediction Error', color='gray', alpha=0.6, linewidth=1)
    ax2.plot(time_axis[plot_range_start:plot_range_end], thr[plot_range_start:plot_range_end], 
             label='Threshold', color='darkorange', linestyle='--', linewidth=1)
    ax2.plot(time_axis[plot_range_start:plot_range_end], -thr[plot_range_start:plot_range_end], 
             color='darkorange', linestyle='--', linewidth=1)
    
    if detected_clicks_indices.size > 0:
        click_times = detected_clicks_indices / sr
        click_errors = err[detected_clicks_indices]
        ax2.scatter(click_times, click_errors, color='red', s=30, zorder=5, 
                   label='Detected Clicks', alpha=0.7)
    
    ax2.set_title('Prediction Error and Detection Threshold')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Error Amplitude')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Convert plot to base64 string
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    plot_data = base64.b64encode(buf.getvalue()).decode('utf-8')
    return plot_data

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_audio():
    try:
        if 'audio_file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['audio_file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not file.filename.lower().endswith('.wav'):
            return jsonify({'error': 'Only WAV files are supported'}), 400
        
        # Get parameters from form
        ar_order = int(request.form.get('ar_order', 4))
        eta = float(request.form.get('eta', 3.0))
        forgetting_factor = float(request.form.get('forgetting_factor', 0.995))
        beta = float(request.form.get('beta', 0.99))
        max_run = int(request.form.get('max_run', 4))
        
        # Save uploaded file temporarily
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
        file.save(input_path)
        
        # Process audio
        samplerate, data = wavfile.read(input_path)
        
        if data.ndim > 1:
            data_mono = data.mean(axis=1)
        else:
            data_mono = data
        
        y_f, dtype, scale = to_float32(data_mono)
        y = y_f.astype(np.float64)
        
        y_clean, a_hist, err, thr, bad_mask = process_audio_online(
            y, order=ar_order, lam=forgetting_factor, eta=eta, 
            beta=beta, delta=1000.0, max_run=max_run
        )
        
        # Generate plot
        plot_data = generate_plot_base64(y, y_clean, samplerate, bad_mask, err, thr)
        
        # Save cleaned audio
        cleaned_audio = from_float32(y_clean.astype(np.float32), dtype, None)
        output_filename = f"cleaned_{file.filename}"
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
        wavfile.write(output_path, samplerate, cleaned_audio)
        
        # Clean up input file
        os.remove(input_path)
        
        # Statistics
        num_clicks = np.sum(bad_mask)
        click_percentage = (num_clicks / len(y)) * 100
        
        return jsonify({
            'success': True,
            'plot': plot_data,
            'download_url': f'/download/{output_filename}',
            'stats': {
                'original_samples': len(y),
                'clicks_detected': int(num_clicks),
                'click_percentage': round(click_percentage, 4),
                'sampling_rate': samplerate
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download/<filename>')
def download_file(filename):
    return send_file(
        os.path.join(app.config['UPLOAD_FOLDER'], filename),
        as_attachment=True,
        download_name=filename
    )

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)