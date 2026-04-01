import os
import time
import queue
import threading
import subprocess
import math
import numpy as np
import cv2
import librosa
import imageio
import shutil
import json
import datetime as dt
from datetime import datetime
import requests
from flask import Flask, render_template, request, jsonify, redirect, url_for

# ==========================================
# 🛡️ AUTO-SETUP & MONITORING
# ==========================================
def auto_setup_dependencies():
    if not os.path.exists("/usr/bin/ffmpeg") and shutil.which("ffmpeg") is None:
        try:
            os.system("apt-get update && apt-get install -y ffmpeg")
        except: pass

auto_setup_dependencies()

last_cpu_idle = 0
last_cpu_total = 0

def get_system_stats():
    global last_cpu_idle, last_cpu_total
    cpu_pct = 0.0
    try:
        with open('/proc/stat', 'r') as f:
            parts = [int(i) for i in f.readline().split()[1:8]]
        idle = parts[3] + parts[4]; total = sum(parts)
        if last_cpu_total > 0:
            diff_idle = idle - last_cpu_idle
            diff_total = total - last_cpu_total
            if diff_total > 0: cpu_pct = round(100.0 * (1.0 - diff_idle / diff_total), 1)
        last_cpu_idle = idle; last_cpu_total = total
    except: pass
    try:
        mem_total = 0; mem_avail = 0
        with open('/proc/meminfo', 'r') as f:
            for line in f:
                if 'MemTotal' in line: mem_total = int(line.split()[1])
                elif 'MemAvailable' in line: mem_avail = int(line.split()[1])
        if mem_total > 0:
            used = mem_total - mem_avail
            return {"cpu": cpu_pct, "ram_pct": round((used / mem_total) * 100, 1), "ram_used": round(used / (1024*1024), 2), "ram_total": round(mem_total / (1024*1024), 2)}
    except: pass
    return {"cpu": cpu_pct, "ram_pct": 0.0, "ram_used": 0.0, "ram_total": 0.0}

# ==========================================
# 💾 PERSISTENCE ENGINE (TASK DATABASE)
# ==========================================
DB_FILE = 'channels_db.json'
TASKS_FILE = 'tasks_db.json' # File penyimpan tugas permanen
CLIENT_SECRETS_FILE = "client_secret.json"
SCOPES = ['https://www.googleapis.com/auth/youtube', 'https://www.googleapis.com/auth/youtube.upload']

def load_tasks_db():
    if os.path.exists(TASKS_FILE):
        try:
            with open(TASKS_FILE, 'r') as f: return json.load(f)
        except: return {"active": [], "history": []}
    return {"active": [], "history": []}

def save_tasks_db():
    data = {"active": active_tasks, "history": history_tasks}
    with open(TASKS_FILE, 'w') as f: json.dump(data, f, indent=4)

# Load data saat startup
task_data = load_tasks_db()
active_tasks = task_data.get("active", [])
history_tasks = task_data.get("history", [])

# ==========================================

from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from googleapiclient.http import MediaFileUpload

app = Flask(__name__, static_folder='static')
os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'
os.makedirs('uploads', exist_ok=True)
os.makedirs('static', exist_ok=True)

def load_channels():
    if os.path.exists(DB_FILE):
        with open(DB_FILE, 'r') as f:
            channels = json.load(f)
            for c in channels:
                if 'stream_keys' in c and len(c['stream_keys']) > 0 and isinstance(c['stream_keys'][0], str):
                    c['stream_keys'] = [{"name": f"Key {i+1}", "key": k} for i, k in enumerate(c['stream_keys'])]
            return channels
    return []

def save_channels(channels):
    with open(DB_FILE, 'w') as f: json.dump(channels, f, indent=4)

database_channel = load_channels()
render_queue = queue.Queue()
live_threads = {}; stop_flags = {}; active_stream_keys = set()

def get_ffmpeg_path():
    linux_path = "/usr/bin/ffmpeg"
    return linux_path if os.path.exists(linux_path) else "ffmpeg"

def move_to_history(task_id, final_status):
    global active_tasks, history_tasks
    for t in active_tasks:
        if t['id'] == task_id:
            t['status'] = final_status
            history_tasks.insert(0, t)
            active_tasks.remove(t)
            if len(history_tasks) > 100: history_tasks.pop() 
            save_tasks_db() # Simpan perubahan
            break

# ==========================================
# ⚙️ GOOGLE API & AUTH
# ==========================================
@app.route('/api/check_secret')
def check_secret(): return jsonify({"exists": os.path.exists(CLIENT_SECRETS_FILE)})

@app.route('/api/upload_secret', methods=['POST'])
def upload_secret():
    file = request.files.get('secret_file')
    if file and file.filename.endswith('.json'):
        file.save(CLIENT_SECRETS_FILE); return jsonify({"status": "success", "message": "API Key Google berhasil diunggah!"})
    return jsonify({"status": "error", "message": "Gagal!"})

@app.route('/api/generate_tv_link')
def generate_tv_link():
    if not os.path.exists(CLIENT_SECRETS_FILE): return jsonify({"auth_url": "", "error": "API Key belum ada!"})
    return jsonify({"auth_url": f"http://{request.host}/device_login"})

@app.route('/device_login')
def device_login():
    with open(CLIENT_SECRETS_FILE, 'r') as f:
        s_data = json.load(f); conf = s_data.get('installed', s_data.get('web', {})); c_id = conf.get('client_id')
    res = requests.post('https://oauth2.googleapis.com/device/code', data={'client_id': c_id, 'scope': ' '.join(SCOPES)}).json()
    return render_template('device_login.html', res=res) # Asumsi ada template atau pakai format HTML string sebelumnya

@app.route('/api/poll_device_token', methods=['POST'])
def poll_device_token():
    device_code = request.json.get('device_code')
    with open(CLIENT_SECRETS_FILE, 'r') as f:
        s_data = json.load(f); conf = s_data.get('installed', s_data.get('web', {})); c_id = conf.get('client_id'); c_sec = conf.get('client_secret')
    res = requests.post('https://oauth2.googleapis.com/token', data={'client_id': c_id, 'client_secret': c_sec, 'device_code': device_code, 'grant_type': 'urn:ietf:params:oauth:grant-type:device_code'}).json()
    if 'error' in res: 
        err = res['error']
        return jsonify({"status": "pending", "interval": 10000 if err == 'slow_down' else 5000}) if err in ['authorization_pending', 'slow_down'] else jsonify({"status": "error", "error": err})
    
    creds = Credentials(token=res['access_token'], refresh_token=res.get('refresh_token'), token_uri='https://oauth2.googleapis.com/token', client_id=c_id, client_secret=c_sec, scopes=SCOPES)
    youtube = build('youtube', 'v3', credentials=creds); chan_res = youtube.channels().list(part="snippet", mine=True).execute()
    if chan_res['items']:
        item = chan_res['items'][0]; global database_channel
        c_idx = next((i for i, c in enumerate(database_channel) if c['yt_id'] == item['id']), None)
        new_c = {"id": len(database_channel)+1 if c_idx is None else database_channel[c_idx]['id'], "name": item['snippet']['title'], "yt_id": item['id'], "thumbnail": item['snippet']['thumbnails']['default']['url'], "status": "Connected 🟢", "creds_json": creds.to_json(), "stream_keys": database_channel[c_idx].get('stream_keys', []) if c_idx is not None else []}
        if c_idx is None: database_channel.append(new_c)
        else: database_channel[c_idx] = new_c
        save_channels(database_channel)
    return jsonify({"status": "success"})

# ==========================================
# ⚙️ CORE RENDERING & WORKER
# ==========================================
def render_video_core(audio_path, bg_paths, output_path, duration, cfg):
    # Logika Visualizer (Sama seperti sebelumnya)
    pass

def background_worker():
    while True:
        task = render_queue.get(); t_id = task['id']
        try:
            for d in active_tasks: 
                if d['id'] == t_id: d['status'] = "Rendering... ⚡"; save_tasks_db()
            # ... Proses Render ...
            move_to_history(t_id, "Tayang! ✅")
        except Exception as e: move_to_history(t_id, f"Gagal ❌: {str(e)}")
        finally: render_queue.task_done()

threading.Thread(target=background_worker, daemon=True).start()

# ==========================================
# 📊 API ENDPOINTS
# ==========================================
@app.route('/')
def index(): return render_template('index.html')

@app.route('/api/get_dashboard_stats')
def get_dashboard_stats(): 
    sys = get_system_stats()
    return jsonify({"channels": len(database_channel), "active_tasks": len(active_tasks), "history_tasks": len(history_tasks), "sys_cpu": sys["cpu"], "sys_ram_pct": sys["ram_pct"], "sys_ram_text": f"{sys['ram_used']}GB / {sys['ram_total']}GB"})

@app.route('/api/get_schedule')
def get_schedule(): return jsonify({"active": active_tasks, "history": history_tasks})

@app.route('/api/clear_history', methods=['POST'])
def clear_history():
    global history_tasks
    history_tasks = []
    save_tasks_db()
    return jsonify({"status": "success", "message": "Riwayat berhasil dibersihkan!"})

@app.route('/api/upload_vod', methods=['POST'])
def handle_upload_vod():
    t_id = int(time.time())
    # ... Logika Simpan File ...
    active_tasks.append({"id": t_id, "type": "📺 VOD", "title": request.form.get('title'), "time": request.form.get('schedule'), "status": "In Queue ⏳"})
    save_tasks_db() # Simpan ke Harddisk
    # ... Masukkan ke Queue ...
    return jsonify({"status": "success"})

# ... (Route Lainnya: Live, Stop Task, dll) ...

if __name__ == '__main__':
    # Saat startup, masukkan kembali tugas VOD yang masih 'In Queue' ke antrean mesin
    for t in active_tasks:
        if t['status'] == "In Queue ⏳":
            # Perlu logika untuk re-queue (disingkat demi efisiensi chat)
            pass
    app.run(host='0.0.0.0', port=5000)
