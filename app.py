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
# 🛡️ AUTO-SETUP DEPENDENCIES & MONITORING MURNI
# ==========================================
def auto_setup_dependencies():
    if not os.path.exists("/usr/bin/ffmpeg") and shutil.which("ffmpeg") is None:
        try:
            print("⚙️ KeiBot: Menginstal FFMPEG secara otomatis...")
            os.system("apt-get update && apt-get install -y ffmpeg")
        except Exception as e: pass

auto_setup_dependencies()

# Menyimpan riwayat CPU untuk perhitungan Real-time
last_cpu_idle = 0
last_cpu_total = 0

def get_system_stats():
    global last_cpu_idle, last_cpu_total
    cpu_pct = 0.0
    # 1. Hitung CPU Real-time murni (Tanpa Library)
    try:
        with open('/proc/stat', 'r') as f:
            parts = [int(i) for i in f.readline().split()[1:8]]
        idle = parts[3] + parts[4]
        total = sum(parts)
        
        if last_cpu_total > 0:
            diff_idle = idle - last_cpu_idle
            diff_total = total - last_cpu_total
            if diff_total > 0:
                cpu_pct = round(100.0 * (1.0 - diff_idle / diff_total), 1)
        
        last_cpu_idle = idle
        last_cpu_total = total
        if cpu_pct < 0.0: cpu_pct = 0.0
        if cpu_pct > 100.0: cpu_pct = 100.0
    except: pass

    # 2. Hitung RAM Real-time murni
    try:
        mem_total = 0; mem_avail = 0
        with open('/proc/meminfo', 'r') as f:
            for line in f:
                if 'MemTotal' in line: mem_total = int(line.split()[1])
                elif 'MemAvailable' in line: mem_avail = int(line.split()[1])
        
        if mem_total > 0:
            used = mem_total - mem_avail
            ram_pct = round((used / mem_total) * 100, 1)
            ram_used_gb = round(used / (1024*1024), 2)
            ram_total_gb = round(mem_total / (1024*1024), 2)
            return {"cpu": cpu_pct, "ram_pct": ram_pct, "ram_used": ram_used_gb, "ram_total": ram_total_gb}
    except: pass
    
    return {"cpu": cpu_pct, "ram_pct": 0.0, "ram_used": 0.0, "ram_total": 0.0}
# ==========================================

from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from googleapiclient.http import MediaFileUpload

app = Flask(__name__, static_folder='static')
os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'
os.makedirs('uploads', exist_ok=True)
os.makedirs('static', exist_ok=True)

DB_FILE = 'channels_db.json'
CLIENT_SECRETS_FILE = "client_secret.json"
SCOPES = ['https://www.googleapis.com/auth/youtube', 'https://www.googleapis.com/auth/youtube.upload']

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
active_tasks = []; history_tasks = []
render_queue = queue.Queue()
live_threads = {}; stop_flags = {}; active_stream_keys = set()

def get_ffmpeg_path():
    local_path = os.path.join(os.path.abspath("."), "ffmpeg.exe")
    if os.path.exists(local_path): return local_path
    linux_path = "/usr/bin/ffmpeg"
    if os.path.exists(linux_path): return linux_path
    return "ffmpeg"

def move_to_history(task_id, final_status):
    global active_tasks, history_tasks
    for t in active_tasks:
        if t['id'] == task_id:
            t['status'] = final_status; history_tasks.insert(0, t); active_tasks.remove(t)
            if len(history_tasks) > 50: history_tasks.pop() 
            break

# ==========================================
# ⚙️ PENGATURAN API GOOGLE (REPARASI DEVICE FLOW)
# ==========================================
@app.route('/api/check_secret')
def check_secret():
    return jsonify({"exists": os.path.exists(CLIENT_SECRETS_FILE)})

@app.route('/api/upload_secret', methods=['POST'])
def upload_secret():
    file = request.files.get('secret_file')
    if file and file.filename.endswith('.json'):
        file.save(CLIENT_SECRETS_FILE)
        return jsonify({"status": "success", "message": "API Key Google berhasil diunggah!"})
    return jsonify({"status": "error", "message": "Gagal! Pastikan file berekstensi .json"})

@app.route('/api/generate_tv_link')
def generate_tv_link():
    if not os.path.exists(CLIENT_SECRETS_FILE): return jsonify({"auth_url": "", "error": "File client_secret.json belum diupload!"})
    return jsonify({"auth_url": f"http://{request.host}/device_login"})

@app.route('/device_login')
def device_login():
    if not os.path.exists(CLIENT_SECRETS_FILE): return "File rahasia tidak ditemukan!"
    with open(CLIENT_SECRETS_FILE, 'r') as f:
        secret_data = json.load(f); client_config = secret_data.get('installed', secret_data.get('web', {})); client_id = client_config.get('client_id')
    res = requests.post('https://oauth2.googleapis.com/device/code', data={'client_id': client_id, 'scope': ' '.join(SCOPES)}).json()
    if 'error' in res: return f"Error Google: {res['error']}"

    # JAVASCRIPT DI BAWAH INI SUDAH DIREPARASI UNTUK MENGHADAPI SLOW_DOWN
    html = f"""
    <html><head><title>Aktivasi YouTube Multi-Profil</title>
    <style>
        body {{ font-family: Arial; text-align: center; background: #1e1e2f; color: white; padding-top: 5vh; }}
        .box {{ background: #2a2a40; width: 550px; margin: auto; padding: 40px; border-radius: 15px; box-shadow: 0 10px 30px rgba(0,0,0,0.5); }}
        .step {{ text-align: left; margin-bottom: 25px; font-size: 16px; color: #ccc; }}
        .input-group {{ display: flex; margin-top: 10px; }}
        .input-group input {{ flex: 1; padding: 15px; font-size: 18px; font-weight: bold; background: #111; color: #00ffcc; border: 1px solid #444; border-radius: 8px 0 0 8px; text-align: center; }}
        .input-group button {{ padding: 15px 25px; font-size: 16px; font-weight: bold; background: #ff0055; color: white; border: none; border-radius: 0 8px 8px 0; cursor: pointer; transition: 0.3s; }}
        .input-group button:hover {{ background: #cc0044; }}
        .status {{ margin-top: 30px; font-size: 16px; color: #aaa; padding: 15px; background: #1a1a2e; border-radius: 8px; border: 1px solid #333; }}
    </style></head><body>
        <div class="box">
            <h2>🔗 Tambah Channel (Multi-Profil)</h2>
            <div class="step">
                <b>Langkah 1:</b> Copy link ini dan <b>Paste di browser / profil Chrome</b> tempat Channel YouTube target Anda berada:
                <div class="input-group">
                    <input type="text" id="glink" value="{res['verification_url']}" readonly>
                    <button onclick="copyTxt('glink', this)">📋 Copy Link</button>
                </div>
            </div>
            <div class="step">
                <b>Langkah 2:</b> Masukkan <b>Kode Rahasia</b> ini di halaman tersebut untuk menyambungkan:
                <div class="input-group">
                    <input type="text" id="gcode" value="{res['user_code']}" readonly>
                    <button onclick="copyTxt('gcode', this)">📋 Copy Kode</button>
                </div>
            </div>
            <div class="status" id="status">⏳ Menunggu Anda memasukkan kode di profil Chrome lain...</div>
        </div>
        <script>
            function copyTxt(id, btn) {{
                var copyText = document.getElementById(id); copyText.select(); document.execCommand("copy");
                var oldTxt = btn.innerHTML; btn.innerHTML = "✅ Copied!"; btn.style.background = "#00cc66";
                setTimeout(() => {{ btn.innerHTML = oldTxt; btn.style.background = "#ff0055"; }}, 2000);
            }}
            function poll() {{
                fetch('/api/poll_device_token', {{
                    method: 'POST', headers: {{'Content-Type': 'application/json'}}, body: JSON.stringify({{device_code: '{res['device_code']}'}})
                }}).then(r => r.json()).then(data => {{
                    if(data.status === 'success') {{
                        document.getElementById('status').innerHTML = "✅ <b>Channel Terhubung!</b> Mengalihkan...";
                        document.getElementById('status').style.color = "#00ffcc"; setTimeout(() => {{ window.location.href = '/'; }}, 2000);
                    }} else if(data.status === 'pending') {{ 
                        // Jika slow_down, script ini akan otomatis melambatkan jeda tunggunya!
                        setTimeout(poll, data.interval || 5000);
                    }} else {{ document.getElementById('status').innerHTML = "❌ Gagal: " + data.error; }}
                }});
            }}
            setTimeout(poll, 5000); // Jeda awal 5 detik sesuai standar Google
        </script>
    </body></html>
    """
    return html

@app.route('/api/poll_device_token', methods=['POST'])
def poll_device_token():
    device_code = request.json.get('device_code')
    with open(CLIENT_SECRETS_FILE, 'r') as f:
        s_data = json.load(f); conf = s_data.get('installed', s_data.get('web', {})); c_id = conf.get('client_id'); c_sec = conf.get('client_secret')
    res = requests.post('https://oauth2.googleapis.com/token', data={'client_id': c_id, 'client_secret': c_sec, 'device_code': device_code, 'grant_type': 'urn:ietf:params:oauth:grant-type:device_code'}).json()
    
    # REPARASI SISTEM PENOLAKAN GOOGLE (SLOW_DOWN)
    if 'error' in res: 
        err = res['error']
        if err == 'authorization_pending':
            return jsonify({"status": "pending", "interval": 5000}) # Tunggu normal 5 detik
        elif err == 'slow_down':
            return jsonify({"status": "pending", "interval": 10000}) # Melambatkan tarikan gas jadi 10 detik agar Google tenang!
        else:
            return jsonify({"status": "error", "error": err}) # Error beneran (seperti expired token)

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

@app.route('/api/save_stream_key', methods=['POST'])
def save_stream_key():
    yt_id = request.form.get('yt_id'); keys_json = request.form.get('stream_keys')
    try: keys_list = json.loads(keys_json)
    except: keys_list = []
    for c in database_channel:
        if c['yt_id'] == yt_id: c['stream_keys'] = keys_list; save_channels(database_channel); return jsonify({"status": "success", "message": "Stream Key diperbarui!"})
    return jsonify({"status": "error", "message": "Channel tidak ditemukan."})

@app.route('/api/get_playlists', methods=['GET'])
def get_playlists():
    yt_id = request.args.get('yt_id')
    if not yt_id: return jsonify([])
    channel = next((c for c in database_channel if c['yt_id'] == yt_id), None)
    if not channel or 'creds_json' not in channel: return jsonify([])
    try:
        creds = Credentials.from_authorized_user_info(json.loads(channel['creds_json'])); youtube = build('youtube', 'v3', credentials=creds)
        res = youtube.playlists().list(part="snippet", mine=True, maxResults=50).execute()
        return jsonify([{"id": p['id'], "title": p['snippet']['title']} for p in res.get('items', [])])
    except: return jsonify([])

@app.route('/api/get_youtube_analytics')
def get_youtube_analytics():
    stats_data = []
    for c in database_channel:
        try:
            creds = Credentials.from_authorized_user_info(json.loads(c['creds_json'])); youtube = build('youtube', 'v3', credentials=creds)
            res = youtube.channels().list(part="statistics", mine=True).execute()
            if res['items']:
                stat = res['items'][0]['statistics']; views = int(stat.get('viewCount', 0)); est_watch_hours = int((views * 3) / 60) 
                stats_data.append({"yt_id": c['yt_id'], "name": c['name'], "thumbnail": c['thumbnail'], "subs": int(stat.get('subscriberCount', 0)), "views": views, "videos": int(stat.get('videoCount', 0)), "watch_hours": est_watch_hours})
        except: pass
    return jsonify(stats_data)

# ==========================================
# ⚙️ CORE ENGINE
# ==========================================
class BackgroundManager:
    def __init__(self, bg_paths, w, h):
        self.bg_paths = bg_paths; self.w = w; self.h = h; self.idx = 0; self.reader = None; self.static_bg = None; self.load_current()
    def load_current(self):
        if self.reader: self.reader.close()
        path = self.bg_paths[self.idx]
        if path.lower().endswith(('.png', '.jpg', '.jpeg')): self.static_bg = cv2.resize(cv2.imread(path), (self.w, self.h))
        else: self.reader = imageio.get_reader(path, 'ffmpeg')
    def get_frame(self):
        if self.static_bg is not None: return self.static_bg.copy()
        try: return cv2.resize(cv2.cvtColor(self.reader.get_next_data(), cv2.COLOR_RGB2BGR), (self.w, self.h))
        except: self.idx = (self.idx + 1) % len(self.bg_paths); self.load_current(); return self.get_frame()
    def close(self):
        if self.reader: self.reader.close()

class AudioBrain:
    def __init__(self): self.y = None; self.sr = None; self.duration = 0.0
    def load(self, path):
        try: self.y, self.sr = librosa.load(path, sr=22050); self.duration = librosa.get_duration(y=self.y, sr=self.sr)
        except: pass
    def get_data(self, t, n_bars=64):
        if self.y is None: return 0.0, False, np.zeros(n_bars)
        idx = int(t * self.sr)
        if idx >= len(self.y): return 0.0, False, np.zeros(n_bars)
        chunk = self.y[idx:idx+1024]; vol = np.sqrt(np.mean(chunk**2)) * 13
        try:
            spec = np.abs(np.fft.rfft(self.y[idx:idx+2048] * np.hanning(2048)))[4:180]
            raw = np.array([np.mean(b) for b in np.array_split(spec, n_bars // 2)]) / 15.0; smooth = np.convolve(raw, np.ones(3)/3, mode='same'); return vol, False, np.concatenate((smooth[::-1], smooth))
        except: return vol, False, np.zeros(n_bars)

class VisualEngine:
    def __init__(self, c_bot, c_top, c_part):
        self.col_bot = (c_bot[2], c_bot[1], c_bot[0]); self.col_top = (c_top[2], c_top[1], c_top[0]); self.col_part = (c_part[2], c_part[1], c_part[0]); self.bar_h = None
        self.grad = np.zeros((1000, 1, 3), dtype=np.uint8)
        for c in range(3): self.grad[:, 0, c] = np.linspace(self.col_top[c], self.col_bot[c], 1000)
        self.particles = []
    def process(self, frame, vol, bars, cfg):
        h, w = frame.shape[:2]; n = len(bars)
        if self.bar_h is None or len(self.bar_h) != n: self.bar_h = np.zeros(n)
        react = float(cfg.get('reactivity', 0.66)); grav = float(cfg.get('gravity', 0.08)); idle = int(cfg.get('idle_height', 5)); space = int(cfg.get('spacing', 3)); px = float(cfg.get('pos_x', 50))/100; py = float(cfg.get('pos_y', 85))/100; wp = float(cfg.get('width_pct', 60))/100; max_h = h * (float(cfg.get('max_height', 40))/100); p_amt = int(cfg.get('part_amount', 3)); p_spd = float(cfg.get('part_speed', 1.0))
        for i in range(n):
            if bars[i] > self.bar_h[i]: self.bar_h[i] = self.bar_h[i]*0.2 + bars[i]*0.8
            else: self.bar_h[i] = max(0, self.bar_h[i] - grav)
        tot_w = w * wp; bar_w = int(max(1, (tot_w - (space * (n-1))) / n)); s_x = int((w * px) - (tot_w / 2)); b_y = int(h * py); mask = np.zeros((h, w), dtype=np.uint8)
        for i in range(n):
            val = self.bar_h[i] * react; height = int(max(idle, min(max_h, val * max_h))); x1 = s_x + (i * (bar_w + space)); x2 = x1 + bar_w; y1 = b_y - height
            if x2 > x1 and y1 < b_y: cv2.rectangle(mask, (x1, y1), (x2, b_y), 255, -1)
        if int(max_h) > 0:
            res = cv2.resize(self.grad, (w, int(max_h))); f_grad = np.zeros((h, w, 3), dtype=np.uint8); y1 = max(0, b_y - int(max_h)); y2 = min(h, b_y); f_grad[y1:y2, :] = res[:y2-y1, :]
            frame = cv2.add(frame, cv2.bitwise_and(f_grad, f_grad, mask=mask))
        if p_amt > 0:
            while len(self.particles) < p_amt: self.particles.append([np.random.randint(0,w), np.random.randint(0,h), np.random.uniform(0.5,2.0), np.random.randint(1,4)])
            while len(self.particles) > p_amt: self.particles.pop()
            for p in self.particles:
                p[1] -= p[2] * p_spd * (1.0 + (vol * 0.1)); 
                if p[1] < 0: p[1] = h; p[0] = np.random.randint(0, w)
                cv2.circle(frame, (int(p[0]), int(p[1])), p[3], self.col_part, -1)
        return frame

def hex_to_rgb(h): return tuple(int(h.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
def render_video_core(audio_path, bg_paths, output_path, duration, cfg):
    w, h = 1280, 720; fps = 30; total_f = int(duration * fps)
    vis = VisualEngine(hex_to_rgb(cfg.get('color_bot')), hex_to_rgb(cfg.get('color_top')), hex_to_rgb(cfg.get('color_part')))
    bg = BackgroundManager(bg_paths, w, h); audio = AudioBrain(); audio.load(audio_path)
    cmd = [get_ffmpeg_path(), '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo', '-s', f'{w}x{h}', '-pix_fmt', 'bgr24', '-r', str(fps), '-i', '-', '-i', audio_path, '-t', str(duration), '-c:v', 'libx264', '-preset', 'fast', '-pix_fmt', 'yuv420p', output_path]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for f in range(total_f):
        v, hit, bars = audio.get_data(f/fps, int(cfg.get('bar_count', 64)))
        proc.stdin.write(vis.process(bg.get_frame(), v, bars, cfg).tobytes())
    proc.stdin.close(); proc.wait(); bg.close()

def background_worker():
    while True:
        task = render_queue.get(); task_id = task['id']
        try:
            if stop_flags.get(task_id): raise Exception("Dibatalkan")
            for d in active_tasks:
                if d['id'] == task_id: d['status'] = "Menyiapkan Base Audio ⚙️"
            base_audio = f"uploads/base_a_{task_id}.mp3"; c_txt = f"uploads/c_{task_id}.txt"
            with open(c_txt, 'w', encoding='utf-8') as f:
                for ap in task['audio_paths']:
                    c_ap = os.path.abspath(ap).replace('\\', '/')
                    f.write(f"file '{c_ap}'\n")
            subprocess.run([get_ffmpeg_path(), '-y', '-f', 'concat', '-safe', '0', '-i', c_txt, '-c', 'copy', base_audio], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            audio = AudioBrain(); audio.load(base_audio); base_dur = audio.duration if audio.duration > 0 else 10
            
            if stop_flags.get(task_id): raise Exception("Dibatalkan")
            for d in active_tasks:
                if d['id'] == task_id: d['status'] = "Rendering Base Video ⚡"
            base_video = f"uploads/base_v_{task_id}.mp4"
            render_video_core(base_audio, task['bg_paths'], base_video, base_dur, task['vis'])
            
            if stop_flags.get(task_id): raise Exception("Dibatalkan")
            loop_count = int(task.get('loop_count', 1)); out_file = f"static/out_{task_id}.mp4"
            if loop_count > 1:
                for d in active_tasks:
                    if d['id'] == task_id: d['status'] = f"Menggandakan Video {loop_count}x 🚀"
                loop_txt = f"uploads/loop_{task_id}.txt"
                with open(loop_txt, 'w', encoding='utf-8') as f:
                    for _ in range(loop_count):
                        c_bv = os.path.abspath(base_video).replace('\\', '/')
                        f.write(f"file '{c_bv}'\n")
                subprocess.run([get_ffmpeg_path(), '-y', '-f', 'concat', '-safe', '0', '-i', loop_txt, '-c', 'copy', out_file], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else: shutil.copy(base_video, out_file)

            meta = task['metadata']; channel_data = next((c for c in database_channel if c['yt_id'] == meta['channel_yt_id']), None)
            if channel_data:
                creds = Credentials.from_authorized_user_info(json.loads(channel_data['creds_json'])); youtube = build('youtube', 'v3', credentials=creds)
                tags_list = [t.strip() for t in meta['tags'].split(',')] if meta['tags'] else []
                sch_raw = meta.get('schedule', ''); sch_obj = datetime.strptime(sch_raw.replace(' ', 'T'), "%Y-%m-%dT%H:%M") if sch_raw else datetime.now()
                body = {'snippet': {'title': meta['title'], 'description': meta['description'], 'tags': tags_list, 'categoryId': '10'}, 'status': {'privacyStatus': 'private'}}
                if sch_obj > datetime.now(): sch_utc = sch_obj - dt.timedelta(hours=7); body['status']['publishAt'] = sch_utc.strftime("%Y-%m-%dT%H:%M:%S.000Z")
                else: body['status']['privacyStatus'] = 'public'

                media = MediaFileUpload(out_file, chunksize=1024*1024*5, resumable=True)
                request_upload = youtube.videos().insert(part=','.join(body.keys()), body=body, media_body=media)
                response_upload = None
                while response_upload is None:
                    if stop_flags.get(task_id): raise Exception("Dibatalkan")
                    status, response_upload = request_upload.next_chunk()
                    if status:
                        for d in active_tasks:
                            if d['id'] == task_id: d['status'] = f"Mengunggah... {int(status.progress() * 100)}% 🚀"

                video_id = response_upload.get('id')
                try:
                    if meta.get('thumbnail_path') and os.path.exists(meta['thumbnail_path']):
                        for d in active_tasks:
                            if d['id'] == task_id: d['status'] = "Memasang Thumbnail... 🖼️"
                        youtube.thumbnails().set(videoId=video_id, media_body=MediaFileUpload(meta['thumbnail_path'])).execute()
                except Exception as e: print("Warning Thumb:", e)
                try:
                    if meta.get('playlist_id'):
                        for d in active_tasks:
                            if d['id'] == task_id: d['status'] = "Menyimpan ke Playlist... 🗂️"
                        youtube.playlistItems().insert(part='snippet', body={'snippet': {'playlistId': meta['playlist_id'], 'resourceId': {'kind': 'youtube#video', 'videoId': video_id}}}).execute()
                except Exception as e: print("Warning Playlist:", e)
                move_to_history(task_id, f"Tayang! ✅ <a href='https://youtu.be/{video_id}' target='_blank'>[Lihat]</a>")
            else: move_to_history(task_id, f"Render Selesai ✅ <a href='/{out_file}' target='_blank'>[Download]</a>")
        except Exception as e: 
            move_to_history(task_id, f"Gagal ❌ (Detail: {str(e)})")
        finally: 
            try: os.remove(f"uploads/base_a_{task_id}.mp3"); os.remove(f"uploads/base_v_{task_id}.mp4")
            except: pass
            render_queue.task_done()

threading.Thread(target=background_worker, daemon=True).start()

def run_live_stream(task_id, stream_key, audio_paths, bg_paths, start_time_str, end_time_str, cfg, metadata):
    try:
        for d in active_tasks:
            if d['id'] == task_id: d['status'] = "Menyiapkan Playlist Audio ⚙️"
        m_audio = f"uploads/live_{task_id}/m.mp3"; c_txt = f"uploads/live_{task_id}/c.txt"
        with open(c_txt, 'w') as f:
            for ap in audio_paths:
                c_ap = os.path.abspath(ap).replace('\\', '/')
                f.write(f"file '{c_ap}'\n")
        subprocess.run([get_ffmpeg_path(), '-y', '-f', 'concat', '-safe', '0', '-i', c_txt, '-c', 'copy', m_audio], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        start_obj = datetime.strptime(start_time_str.replace('T', ' '), "%Y-%m-%d %H:%M")
        while datetime.now() < start_obj:
            if stop_flags.get(task_id): raise Exception("Dibatalkan")
            for d in active_tasks:
                if d['id'] == task_id: d['status'] = f"Menunggu Jadwal Mulai ⏳ ({start_time_str})"
            time.sleep(2)

        for d in active_tasks:
            if d['id'] == task_id: d['status'] = "Memperbarui Metadata Live... 📡"
        channel_data = next((c for c in database_channel if c['yt_id'] == metadata['channel_yt_id']), None)
        
        if channel_data:
            try:
                creds = Credentials.from_authorized_user_info(json.loads(channel_data['creds_json'])); youtube = build('youtube', 'v3', credentials=creds)
                live_res = youtube.liveBroadcasts().list(part="snippet", broadcastStatus="active", broadcastType="all").execute()
                if not live_res.get('items'):
                    live_res = youtube.liveBroadcasts().list(part="snippet", broadcastStatus="upcoming", broadcastType="all").execute()
                
                if live_res.get('items'):
                    b_id = live_res['items'][0]['id']
                    video_res = youtube.videos().list(part="snippet", id=b_id).execute()
                    if video_res.get('items'):
                        v_snip = video_res['items'][0]['snippet']
                        v_snip['title'] = metadata['title']
                        v_snip['description'] = metadata['description']
                        youtube.videos().update(part="snippet", body={"id": b_id, "snippet": v_snip}).execute()
                    
                    if metadata.get('thumbnail_path') and os.path.exists(metadata['thumbnail_path']):
                        youtube.thumbnails().set(videoId=b_id, media_body=MediaFileUpload(metadata['thumbnail_path'])).execute()
            except Exception as e: print("Live API Metadata Error:", e) 

        for d in active_tasks:
            if d['id'] == task_id: d['status'] = "ON AIR (LIVE) 🔴"
        rtmp_url = f"rtmp://a.rtmp.youtube.com/live2/{stream_key}"; vis = VisualEngine(hex_to_rgb(cfg.get('color_bot')), hex_to_rgb(cfg.get('color_top')), hex_to_rgb(cfg.get('color_part'))); bg = BackgroundManager(bg_paths, 1280, 720); audio = AudioBrain(); audio.load(m_audio)
        cmd = [get_ffmpeg_path(), '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo', '-s', '1280x720', '-pix_fmt', 'bgr24', '-r', '30', '-i', '-', '-stream_loop', '-1', '-i', m_audio, '-c:v', 'libx264', '-preset', 'veryfast', '-b:v', '2500k', '-maxrate', '2500k', '-bufsize', '5000k', '-pix_fmt', 'yuv420p', '-g', '60', '-c:a', 'aac', '-b:a', '128k', '-f', 'flv', rtmp_url]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE); live_threads[task_id] = proc
        
        f_idx = 0; end_obj = datetime.strptime(end_time_str.replace('T', ' '), "%Y-%m-%d %H:%M")
        while True:
            if stop_flags.get(task_id) or datetime.now() >= end_obj: break
            v, hit, bars = audio.get_data((f_idx/30) % audio.duration if audio.duration > 0 else 0, int(cfg.get('bar_count', 64)))
            proc.stdin.write(vis.process(bg.get_frame(), v, bars, cfg).tobytes()); f_idx += 1
            
        proc.terminate(); bg.close(); shutil.rmtree(f"uploads/live_{task_id}", ignore_errors=True); active_stream_keys.discard(stream_key) 
        if stop_flags.get(task_id): move_to_history(task_id, "Dihentikan Paksa ⏹️")
        else: move_to_history(task_id, "Live Selesai 🧹")
    except Exception as e:
        active_stream_keys.discard(stream_key)
        move_to_history(task_id, f"Live Gagal ❌ (Detail: {str(e)})")

# ==========================================
# 📊 API ENDPOINTS
# ==========================================
@app.route('/')
def index(): return render_template('index.html')

@app.route('/api/get_dashboard_stats')
def get_dashboard_stats(): 
    sys = get_system_stats()
    return jsonify({
        "channels": len(database_channel), "active_tasks": len(active_tasks), "history_tasks": len(history_tasks),
        "sys_cpu": sys["cpu"], "sys_ram_pct": sys["ram_pct"], "sys_ram_text": f"{sys['ram_used']}GB / {sys['ram_total']}GB"
    })

@app.route('/api/get_schedule')
def get_schedule(): return jsonify({"active": active_tasks, "history": history_tasks})

@app.route('/api/get_channels')
def get_channels():
    safe_c = [{"id": c["id"], "name": c["name"], "yt_id": c["yt_id"], "thumbnail": c["thumbnail"], "status": c["status"], "stream_keys": c.get("stream_keys", [])} for c in database_channel]
    return jsonify(safe_c)

@app.route('/api/stop_task/<int:task_id>', methods=['POST'])
def stop_task(task_id):
    stop_flags[task_id] = True
    if task_id in live_threads:
        try: live_threads[task_id].terminate()
        except: pass
    return jsonify({"status": "success", "message": "Dihentikan!"})

@app.route('/api/preview_visualizer', methods=['POST'])
def preview_visualizer():
    try:
        audios = request.files.getlist('audios'); bgs = request.files.getlist('background'); a_p = "uploads/p.mp3"; v_p = "uploads/p_bg" + os.path.splitext(bgs[0].filename)[1]; audios[0].save(a_p); bgs[0].save(v_p)
        render_video_core(a_p, [v_p], "static/p.mp4", 5.0, request.form)
        return jsonify({"status": "success", "preview_url": "/static/p.mp4?t="+str(time.time())})
    except Exception as e: return jsonify({"status": "error", "message": str(e)})

@app.route('/api/upload_vod', methods=['POST'])
def handle_upload_vod():
    t_id = int(time.time()); audios = request.files.getlist('audios'); bgs = request.files.getlist('background'); a_ps = []; v_ps = []
    for i, a in enumerate(audios): 
        if a.filename: p = f"uploads/vod_a_{t_id}_{i}.mp3"; a.save(p); a_ps.append(p)
    for i, b in enumerate(bgs): 
        if b.filename: p = f"uploads/vod_v_{t_id}_{i}{os.path.splitext(b.filename)[1]}"; b.save(p); v_ps.append(p)
    thumb_file = request.files.get('thumbnail'); thumb_path = ""
    if thumb_file and thumb_file.filename: thumb_path = f"uploads/vod_thumb_{t_id}{os.path.splitext(thumb_file.filename)[1]}"; thumb_file.save(thumb_path)

    metadata = {"channel_yt_id": request.form.get('channel_select', ''), "title": request.form.get('title', ''), "description": request.form.get('description', ''), "tags": request.form.get('tags', ''), "playlist_id": request.form.get('playlist', ''), "thumbnail_path": thumb_path, "schedule": request.form.get('schedule', '')}
    loop_count = int(request.form.get('loop_count', 1))
    active_tasks.append({"id": t_id, "type": "📺 VOD", "title": metadata['title'], "time": request.form.get('schedule').replace('T',' '), "status": "In Queue ⏳"})
    render_queue.put({"id": t_id, "audio_paths": a_ps, "bg_paths": v_ps, "vis": request.form, "loop_count": loop_count, "metadata": metadata})
    return jsonify({"status": "success", "message": "Masuk Antrean VOD!"})

@app.route('/api/schedule_live', methods=['POST'])
def handle_schedule_live():
    stream_key = request.form.get('stream_key')
    if not stream_key: return jsonify({"status": "error", "message": "Harap pilih Stream Key dari menu Dropdown!"})
    if stream_key in active_stream_keys: return jsonify({"status": "error", "message": "Stream Key ini SEDANG DIPAKAI oleh tugas Live lain! Silakan pilih Key yang berbeda."})
    
    active_stream_keys.add(stream_key)
    yt_id = request.form.get('channel_select')
    
    t_id = int(time.time()); os.makedirs(f"uploads/live_{t_id}", exist_ok=True)
    audios = request.files.getlist('audios'); bgs = request.files.getlist('background'); a_ps = []; v_ps = []
    for i, a in enumerate(audios): 
        if a.filename: p = f"uploads/live_{t_id}/a_{i}.mp3"; a.save(p); a_ps.append(p)
    for i, b in enumerate(bgs): 
        if b.filename: p = f"uploads/live_{t_id}/v_{i}{os.path.splitext(b.filename)[1]}"; b.save(p); v_ps.append(p)
    thumb_file = request.files.get('thumbnail'); thumb_path = ""
    if thumb_file and thumb_file.filename: thumb_path = f"uploads/live_{t_id}/thumb{os.path.splitext(thumb_file.filename)[1]}"; thumb_file.save(thumb_path)

    metadata = {"channel_yt_id": yt_id, "title": request.form.get('title', ''), "description": request.form.get('description', ''), "tags": request.form.get('tags', ''), "thumbnail_path": thumb_path}
    
    active_tasks.append({"id": t_id, "type": "🔴 LIVE", "title": metadata['title'], "time": f"Mulai: {request.form.get('schedule_start').replace('T', ' ')}", "status": "In Queue ⏳"})
    threading.Thread(target=run_live_stream, args=(t_id, stream_key, a_ps, v_ps, request.form.get('schedule_start'), request.form.get('schedule_end'), request.form, metadata)).start()
    return jsonify({"status": "success", "message": "Live Engine Dijadwalkan!"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
