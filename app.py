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
# 🛡️ AUTO-SETUP DEPENDENCIES
# ==========================================
def auto_setup_dependencies():
    if not os.path.exists("/usr/bin/ffmpeg") and shutil.which("ffmpeg") is None:
        try:
            print("⚙️ KeiBot: Menginstal FFMPEG secara otomatis...")
            os.system("apt-get update && apt-get install -y ffmpeg")
        except Exception as e: pass

auto_setup_dependencies()

last_cpu_idle = 0
last_cpu_total = 0

def get_system_stats():
    global last_cpu_idle, last_cpu_total
    cpu_pct = 0.0
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
# 💾 DATABASE ENGINE
# ==========================================
DB_FILE = 'channels_db.json'
TASKS_FILE = 'tasks_db.json'
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

task_data = load_tasks_db()
active_tasks = task_data.get("active", [])
history_tasks = task_data.get("history", [])

# ==========================================
# 🚦 RAM GATEKEEPER
# ==========================================
def wait_for_resources(task_id, max_ram_pct=85.0):
    while True:
        if stop_flags.get(task_id): return False
        stats = get_system_stats()
        if stats['ram_pct'] < max_ram_pct: return True
        for d in active_tasks:
            if d['id'] == task_id: d['status'] = f"Menunggu RAM Turun ({stats['ram_pct']}%) ⏳"
        save_tasks_db()
        time.sleep(10)

# ==========================================
# ⚙️ INIT FLASK & CORE ENGINE
# ==========================================
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from googleapiclient.http import MediaFileUpload

app = Flask(__name__, static_folder='static')
os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'
os.makedirs('uploads', exist_ok=True)
os.makedirs('static', exist_ok=True)

def load_channels():
    if os.path.exists(DB_FILE):
        with open(DB_FILE, 'r') as f: return json.load(f)
    return []

def save_channels(channels):
    with open(DB_FILE, 'w') as f: json.dump(channels, f, indent=4)

database_channel = load_channels()
render_queue = queue.Queue()
live_threads = {}; stop_flags = {}; active_stream_keys = set()

def get_ffmpeg_path():
    local_path = os.path.join(os.path.abspath("."), "ffmpeg.exe")
    if os.path.exists(local_path): return local_path
    return "/usr/bin/ffmpeg" if os.path.exists("/usr/bin/ffmpeg") else "ffmpeg"

def move_to_history(task_id, final_status):
    global active_tasks, history_tasks
    for t in active_tasks:
        if t['id'] == task_id:
            t['status'] = final_status; history_tasks.insert(0, t); active_tasks.remove(t)
            if len(history_tasks) > 50: history_tasks.pop() 
            save_tasks_db()
            break

class AudioBrain:
    def __init__(self): self.y = None; self.sr = None; self.duration = 0.0
    def load(self, path, max_duration=None):
        try: 
            self.y, self.sr = librosa.load(path, sr=22050, duration=max_duration)
            self.duration = librosa.get_duration(path=path) 
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

class VisualEngine:
    def __init__(self, c_bot, c_top, c_part):
        self.col_bot = (c_bot[2], c_bot[1], c_bot[0]); self.col_top = (c_top[2], c_top[1], c_top[0]); self.col_part = (c_part[2], c_part[1], c_part[0]); self.bar_h = None
        self.grad = np.zeros((1000, 1, 3), dtype=np.uint8)
        for c in range(3): self.grad[:, 0, c] = np.linspace(self.col_top[c], self.col_bot[c], 1000)
        self.particles = []
    def process(self, frame, vol, bars, cfg):
        h, w = frame.shape[:2]; n = len(bars)
        if self.bar_h is None or len(self.bar_h) != n: self.bar_h = np.zeros(n)
        def safe_num(val, default):
            try: return float(val) if val != "" and val is not None else default
            except: return default
        react = safe_num(cfg.get('reactivity'), 0.66); grav = safe_num(cfg.get('gravity'), 0.08)
        idle = int(safe_num(cfg.get('idle_height'), 5)); space = int(safe_num(cfg.get('spacing'), 3))
        px = safe_num(cfg.get('pos_x'), 50)/100; py = safe_num(cfg.get('pos_y'), 85)/100
        wp = safe_num(cfg.get('width_pct'), 60)/100; max_h = h * (safe_num(cfg.get('max_height'), 40)/100)
        p_amt = int(safe_num(cfg.get('part_amount'), 3)); p_spd = safe_num(cfg.get('part_speed'), 1.0)
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
    cmd = [get_ffmpeg_path(), '-y', '-threads', '2', '-f', 'rawvideo', '-vcodec', 'rawvideo', '-s', f'{w}x{h}', '-pix_fmt', 'bgr24', '-r', str(fps), '-i', '-', '-i', audio_path, '-t', str(duration), '-c:v', 'libx264', '-preset', 'fast', '-pix_fmt', 'yuv420p', output_path]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for f in range(total_f):
        v, _, bars = audio.get_data(f/fps, int(cfg.get('bar_count', 64)))
        proc.stdin.write(vis.process(bg.get_frame(), v, bars, cfg).tobytes())
    proc.stdin.close(); proc.wait(); bg.close()

def background_worker():
    while True:
        task = render_queue.get(); task_id = task['id']
        try:
            if not wait_for_resources(task_id): raise Exception("RAM Kepenuhan")
            for d in active_tasks:
                if d['id'] == task_id: d['status'] = "Menyiapkan Audio ⚙️"
            save_tasks_db()
            
            base_audio = f"uploads/base_a_{task_id}.mp3"; c_txt = f"uploads/c_{task_id}.txt"
            with open(c_txt, 'w', encoding='utf-8') as f:
                for ap in task['audio_paths']:
                    safe_p = os.path.abspath(ap).replace('\\', '/')
                    f.write(f"file '{safe_p}'\n")
            subprocess.run([get_ffmpeg_path(), '-y', '-threads', '2', '-f', 'concat', '-safe', '0', '-i', c_txt, '-c', 'copy', base_audio])
            
            audio = AudioBrain(); audio.load(base_audio); base_dur = audio.duration if audio.duration > 0 else 10
            base_video = f"uploads/base_v_{task_id}.mp4"
            
            for d in active_tasks:
                if d['id'] == task_id: d['status'] = "Rendering Video ⚡"
            save_tasks_db()
            render_video_core(base_audio, task['bg_paths'], base_video, base_dur, task['vis'])
            
            loop_count = int(task.get('loop_count', 1)); out_file = f"static/out_{task_id}.mp4"
            if loop_count > 1:
                for d in active_tasks:
                    if d['id'] == task_id: d['status'] = f"Menggandakan {loop_count}x 🚀"
                save_tasks_db()
                loop_txt = f"uploads/loop_{task_id}.txt"
                with open(loop_txt, 'w', encoding='utf-8') as f:
                    for _ in range(loop_count):
                        safe_bv = os.path.abspath(base_video).replace('\\', '/')
                        f.write(f"file '{safe_bv}'\n")
                subprocess.run([get_ffmpeg_path(), '-y', '-threads', '2', '-f', 'concat', '-safe', '0', '-i', loop_txt, '-c', 'copy', out_file])
            else: shutil.copy(base_video, out_file)

            meta = task['metadata']; channel_data = next((c for c in database_channel if c['yt_id'] == meta['channel_yt_id']), None)
            if channel_data:
                creds = Credentials.from_authorized_user_info(json.loads(channel_data['creds_json'])); youtube = build('youtube', 'v3', credentials=creds)
                tags_list = [t.strip() for t in meta['tags'].split(',')] if meta['tags'] else []
                sch_raw = meta.get('schedule', ''); sch_obj = datetime.strptime(sch_raw.replace(' ', 'T'), "%Y-%m-%dT%H:%M") if sch_raw else datetime.now()
                
                body = {'snippet': {'title': meta['title'], 'description': meta['description'], 'tags': tags_list, 'categoryId': '10'}, 
                        'status': {'privacyStatus': meta.get('privacy', 'public')}}
                
                if sch_obj > datetime.now():
                    sch_utc = sch_obj - dt.timedelta(hours=7)
                    body['status']['publishAt'] = sch_utc.strftime("%Y-%m-%dT%H:%M:%S.000Z")
                    body['status']['privacyStatus'] = 'private'

                media = MediaFileUpload(out_file, chunksize=1024*1024*5, resumable=True)
                req = youtube.videos().insert(part=','.join(body.keys()), body=body, media_body=media)
                resp = None
                while resp is None:
                    status, resp = req.next_chunk()
                    if status:
                        for d in active_tasks:
                            if d['id'] == task_id: d['status'] = f"Mengunggah {int(status.progress()*100)}% 🚀"
                        save_tasks_db()
                
                video_id = resp.get('id')
                if meta.get('thumbnail_path') and os.path.exists(meta['thumbnail_path']):
                    youtube.thumbnails().set(videoId=video_id, media_body=MediaFileUpload(meta['thumbnail_path'])).execute()
                if meta.get('playlist_id'):
                    youtube.playlistItems().insert(part='snippet', body={'snippet': {'playlistId': meta['playlist_id'], 'resourceId': {'kind': 'youtube#video', 'videoId': video_id}}}).execute()
                
                move_to_history(task_id, f"Tayang! ✅ <a href='https://youtu.be/{video_id}' target='_blank'>[Lihat]</a>")
            else: move_to_history(task_id, f"Render Selesai ✅ <a href='/{out_file}' target='_blank'>[Download]</a>")
        except Exception as e: move_to_history(task_id, f"Gagal ❌ ({str(e)})")
        finally: 
            try: os.remove(f"uploads/base_a_{task_id}.mp3"); os.remove(f"uploads/base_v_{task_id}.mp4")
            except: pass
            render_queue.task_done()

threading.Thread(target=background_worker, daemon=True).start()

def run_live_stream(task_id, stream_key, audio_paths, bg_paths, start_time_str, end_time_str, cfg, metadata):
    try:
        if not wait_for_resources(task_id): raise Exception("RAM Kepenuhan")
        m_audio = f"uploads/live_{task_id}/m.mp3"; c_txt = f"uploads/live_{task_id}/c.txt"
        os.makedirs(f"uploads/live_{task_id}", exist_ok=True)
        with open(c_txt, 'w', encoding='utf-8') as f:
            for ap in audio_paths:
                safe_pl = os.path.abspath(ap).replace('\\', '/')
                f.write(f"file '{safe_pl}'\n")
        subprocess.run([get_ffmpeg_path(), '-y', '-threads', '2', '-f', 'concat', '-safe', '0', '-i', c_txt, '-c', 'copy', m_audio])
        
        start_obj = datetime.strptime(start_time_str.replace('T', ' '), "%Y-%m-%d %H:%M")
        while datetime.now() < start_obj:
            if stop_flags.get(task_id): raise Exception("Dibatalkan")
            for d in active_tasks:
                if d['id'] == task_id: d['status'] = f"Menunggu Jadwal ⏳ ({start_time_str})"
            save_tasks_db(); time.sleep(10)

        channel_data = next((c for c in database_channel if c['yt_id'] == metadata['channel_yt_id']), None)
        if channel_data:
            try:
                creds = Credentials.from_authorized_user_info(json.loads(channel_data['creds_json'])); youtube = build('youtube', 'v3', credentials=creds)
                live_res = youtube.liveBroadcasts().list(part="snippet,status", broadcastStatus="active", broadcastType="all").execute()
                if not live_res.get('items'): live_res = youtube.liveBroadcasts().list(part="snippet,status", broadcastStatus="upcoming", broadcastType="all").execute()
                if live_res.get('items'):
                    b_id = live_res['items'][0]['id']
                    v_snip = live_res['items'][0]['snippet']
                    v_stat = live_res['items'][0]['status']
                    v_snip['title'] = metadata['title']; v_snip['description'] = metadata['description']
                    v_stat['privacyStatus'] = metadata.get('privacy', 'public')
                    youtube.videos().update(part="snippet,status", body={"id": b_id, "snippet": v_snip, "status": v_stat}).execute()
            except Exception as e: print("Live API Metadata Error:", e) 

        for d in active_tasks:
            if d['id'] == task_id: d['status'] = "ON AIR (LIVE) 🔴"
        save_tasks_db()

        rtmp_url = f"rtmp://a.rtmp.youtube.com/live2/{stream_key}"
        vis = VisualEngine(hex_to_rgb(cfg.get('color_bot')), hex_to_rgb(cfg.get('color_top')), hex_to_rgb(cfg.get('color_part')))
        bg = BackgroundManager(bg_paths, 1280, 720); audio = AudioBrain(); audio.load(m_audio, max_duration=600)
        
        cmd = [get_ffmpeg_path(), '-y', '-threads', '2', '-f', 'rawvideo', '-vcodec', 'rawvideo', '-s', '1280x720', '-pix_fmt', 'bgr24', '-r', '30', '-i', '-', '-stream_loop', '-1', '-i', m_audio, '-c:v', 'libx264', '-preset', 'veryfast', '-b:v', '2500k', '-f', 'flv', rtmp_url]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE); live_threads[task_id] = proc
        
        f_idx = 0; end_obj = datetime.strptime(end_time_str.replace('T', ' '), "%Y-%m-%d %H:%M")
        while not stop_flags.get(task_id) and datetime.now() < end_obj:
            v, _, bars = audio.get_data((f_idx/30) % audio.duration if audio.duration > 0 else 0, int(cfg.get('bar_count', 64)))
            proc.stdin.write(vis.process(bg.get_frame(), v, bars, cfg).tobytes()); f_idx += 1
            
        proc.terminate(); bg.close(); active_stream_keys.discard(stream_key)
        move_to_history(task_id, "Live Selesai 🧹")
    except Exception as e:
        active_stream_keys.discard(stream_key)
        move_to_history(task_id, f"Gagal ❌ ({str(e)})")

# ==========================================
# 📊 API ENDPOINTS
# ==========================================
@app.route('/')
def index(): return render_template('index.html')

@app.route('/api/get_dashboard_stats')
def get_dashboard_stats():
    sys = get_system_stats()
    return jsonify({"channels": len(database_channel), "active_tasks": len(active_tasks), "history_tasks": len(history_tasks),
                    "sys_cpu": sys["cpu"], "sys_ram_pct": sys["ram_pct"], "sys_ram_text": f"{sys['ram_used']}GB / {sys['ram_total']}GB"})

@app.route('/api/get_schedule')
def get_schedule(): return jsonify({"active": active_tasks, "history": history_tasks})

@app.route('/api/clear_history', methods=['POST'])
def clear_history():
    global history_tasks; history_tasks = []; save_tasks_db()
    return jsonify({"status": "success", "message": "Riwayat dibersihkan!"})

@app.route('/api/get_channels')
def get_channels():
    safe_c = [{"id": c["id"], "name": c["name"], "yt_id": c["yt_id"], "thumbnail": c["thumbnail"], "status": c["status"], 
               "stream_keys": c.get("stream_keys", []), "title_bank": c.get("title_bank", [])} for c in database_channel]
    return jsonify(safe_c)

@app.route('/api/delete_channel', methods=['POST'])
def delete_channel():
    yt_id = request.form.get('yt_id'); global database_channel
    database_channel = [c for c in database_channel if c['yt_id'] != yt_id]; save_channels(database_channel)
    return jsonify({"status": "success", "message": "Channel dihapus!"})

@app.route('/api/upload_title_bank', methods=['POST'])
def upload_title_bank():
    yt_id = request.form.get('yt_id'); file = request.files.get('txt_file')
    if not yt_id or not file: return jsonify({"status": "error", "message": "Bahan kurang!"})
    try:
        lines = [l.strip() for l in file.read().decode('utf-8').split('\n') if l.strip()]
        global database_channel
        for c in database_channel:
            if c['yt_id'] == yt_id: c['title_bank'] = lines; save_channels(database_channel); return jsonify({"status": "success", "message": f"Tersimpan {len(lines)} judul!"})
        return jsonify({"status": "error", "message": "Channel tidak ketemu."})
    except Exception as e: return jsonify({"status": "error", "message": str(e)})

@app.route('/api/upload_vod', methods=['POST'])
def handle_upload_vod():
    t_id = int(time.time()); audios = request.files.getlist('audios'); bgs = request.files.getlist('background'); a_ps = []; v_ps = []
    for i, a in enumerate(audios): p = f"uploads/a_{t_id}_{i}.mp3"; a.save(p); a_ps.append(p)
    for i, b in enumerate(bgs): p = f"uploads/b_{t_id}_{i}.mp4"; b.save(p); v_ps.append(p)
    thumb = request.files.get('thumbnail'); t_path = ""
    if thumb: t_path = f"uploads/t_{t_id}.jpg"; thumb.save(t_path)
    
    metadata = {"channel_yt_id": request.form.get('channel_select', ''), "title": request.form.get('title', ''), 
                "description": request.form.get('description', ''), "tags": request.form.get('tags', ''), 
                "playlist_id": request.form.get('playlist', ''), "thumbnail_path": t_path, 
                "schedule": request.form.get('schedule', ''), "privacy": request.form.get('privacy', 'public')}
    
    active_tasks.append({"id": t_id, "type": "📺 VOD", "title": metadata['title'], "time": metadata['schedule'].replace('T',' ') if metadata['schedule'] else "Instan", "status": "In Queue ⏳"})
    save_tasks_db(); render_queue.put({"id": t_id, "audio_paths": a_ps, "bg_paths": v_ps, "vis": dict(request.form), "loop_count": int(request.form.get('loop_count', 1)), "metadata": metadata})
    return jsonify({"status": "success"})

@app.route('/api/schedule_live', methods=['POST'])
def handle_schedule_live():
    sk = request.form.get('stream_key'); yt_id = request.form.get('channel_select'); st = request.form.get('schedule_start'); et = request.form.get('schedule_end')
    if not sk or not st: return jsonify({"status": "error", "message": "Lengkapi form!"})
    
    t_id = int(time.time()); audios = request.files.getlist('audios'); bgs = request.files.getlist('background'); a_ps = []; v_ps = []
    for i, a in enumerate(audios): p = f"uploads/la_{t_id}_{i}.mp3"; a.save(p); a_ps.append(p)
    for i, b in enumerate(bgs): p = f"uploads/lb_{t_id}_{i}.mp4"; b.save(p); v_ps.append(p)

    metadata = {"channel_yt_id": yt_id, "title": request.form.get('title', ''), "description": request.form.get('description', ''), "privacy": request.form.get('privacy', 'public')}
    active_tasks.append({"id": t_id, "type": "🔴 LIVE", "title": metadata['title'], "time": st.replace('T',' '), "status": "In Queue ⏳"})
    save_tasks_db(); threading.Thread(target=run_live_stream, args=(t_id, sk, a_ps, v_ps, st, et, dict(request.form), metadata)).start()
    return jsonify({"status": "success"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
