#!/usr/bin/env python3
"""Фотограмметрийн локал вэб апп.

Зөвхөн Python-ий стандарт сангуудыг ашигладаг (numpy/cv2 шаардахгүй).
Ажиллуулах:  python3 server.py  →  http://localhost:8123
"""
import http.server, socketserver, json, os, re, shutil, subprocess, threading, time, uuid, urllib.parse

ROOT = os.path.dirname(os.path.abspath(__file__))
JOBS = os.path.join(ROOT, 'jobs')
WEB  = os.path.join(ROOT, 'web')
PORT = int(os.environ.get('PORT', '8123'))
os.makedirs(JOBS, exist_ok=True)

SAFE = re.compile(r'[^A-Za-z0-9._-]')
def safe_name(n): return SAFE.sub('_', n)[-120:] or 'file'
def job_dir(jid):
    if not re.fullmatch(r'[a-f0-9]{12}', jid or ''): return None
    return os.path.join(JOBS, jid)

_procs = {}   # jid -> Popen

FEATURE_MODES = ('sift', 'sift-hard', 'aliked')

def start_job(jid, features='sift'):
    d = job_dir(jid)
    log = open(os.path.join(d, 'log.txt'), 'ab', buffering=0)
    env = dict(os.environ)
    env['FEATURES'] = features if features in FEATURE_MODES else 'sift'
    p = subprocess.Popen(['bash', os.path.join(ROOT, 'recon.sh'), d],
                         stdout=log, stderr=subprocess.STDOUT, cwd=ROOT, env=env)
    _procs[jid] = p
    def wait():
        rc = p.wait()
        with open(os.path.join(d, 'exit'), 'w') as f: f.write(str(rc))
    threading.Thread(target=wait, daemon=True).start()

def job_state(jid):
    d = job_dir(jid)
    if not d or not os.path.isdir(d): return None
    logp = os.path.join(d, 'log.txt')
    text = ''
    if os.path.exists(logp):
        with open(logp, 'rb') as f:
            f.seek(0, 2); size = f.tell()
            f.seek(max(0, size - 20000))
            text = f.read().decode('utf-8', 'replace')
    stages = [int(m) for m in re.findall(r'@@STAGE (\d+)', text)]
    exitp = os.path.join(d, 'exit')
    rc = None
    if os.path.exists(exitp):
        rc = int(open(exitp).read().strip() or '1')
    imgs = os.path.join(d, 'images')
    outs = {}
    for name in ('dense/fused.ply', 'dense/object.ply', 'dense/mesh.ply'):
        p = os.path.join(d, name)
        if os.path.exists(p): outs[name] = os.path.getsize(p)
    running = jid in _procs and _procs[jid].poll() is None
    return { 'id':jid, 'running':running, 'exit':rc,
             'stage':(stages[-1] if stages else 0),
             'images':len(os.listdir(imgs)) if os.path.isdir(imgs) else 0,
             'outputs':outs,
             'log':'\n'.join(l for l in text.splitlines() if not l.startswith('@@'))[-6000:] }

class H(http.server.BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'
    def log_message(self, *a): pass

    def _send(self, code, body=b'', ctype='application/json', extra=None):
        if isinstance(body, str): body = body.encode()
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        for k, v in (extra or {}).items(): self.send_header(k, v)
        self.end_headers()
        if self.command != 'HEAD': self.wfile.write(body)

    def _json(self, code, obj): self._send(code, json.dumps(obj), 'application/json')

    # ---------- GET ----------
    def do_GET(self):
        path = urllib.parse.urlparse(self.path).path
        if path == '/' or path == '/index.html':
            return self._file(os.path.join(WEB, 'index.html'), 'text/html; charset=utf-8')
        if path == '/api/jobs':
            ids = sorted((d for d in os.listdir(JOBS) if job_dir(d)),
                         key=lambda d: os.path.getmtime(os.path.join(JOBS, d)), reverse=True)
            return self._json(200, [job_state(i) for i in ids[:20]])
        m = re.fullmatch(r'/api/job/([a-f0-9]{12})/status', path)
        if m:
            st = job_state(m.group(1))
            return self._json(200, st) if st else self._json(404, {'error':'олдсонгүй'})
        m = re.fullmatch(r'/api/job/([a-f0-9]{12})/file/(dense/(?:fused|object|mesh)\.ply)', path)
        if m:
            d = job_dir(m.group(1))
            return self._file(os.path.join(d, m.group(2)), 'application/octet-stream')
        return self._send(404, b'not found', 'text/plain')

    def _file(self, p, ctype):
        if not os.path.exists(p): return self._send(404, b'not found', 'text/plain')
        size = os.path.getsize(p)
        self.send_response(200)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(size))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        with open(p, 'rb') as f: shutil.copyfileobj(f, self.wfile)

    # ---------- POST / PUT ----------
    def do_POST(self):
        path = urllib.parse.urlparse(self.path).path
        if path == '/api/job':
            jid = uuid.uuid4().hex[:12]
            os.makedirs(os.path.join(JOBS, jid, 'images'), exist_ok=True)
            return self._json(200, {'id':jid})
        m = re.fullmatch(r'/api/job/([a-f0-9]{12})/run', path)
        if m:
            jid = m.group(1)
            d = job_dir(jid)
            if not d or not os.path.isdir(d): return self._json(404, {'error':'олдсонгүй'})
            if jid in _procs and _procs[jid].poll() is None:
                return self._json(409, {'error':'аль хэдийн ажиллаж байна'})
            for f in ('log.txt', 'exit'):
                try: os.remove(os.path.join(d, f))
                except FileNotFoundError: pass
            q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            feat = (q.get('features') or ['sift'])[0]
            start_job(jid, feat)
            return self._json(200, {'ok':True, 'features':feat})
        return self._send(404, b'not found', 'text/plain')

    def do_PUT(self):
        m = re.fullmatch(r'/api/job/([a-f0-9]{12})/img/(.+)', urllib.parse.urlparse(self.path).path)
        if not m: return self._send(404, b'not found', 'text/plain')
        d = job_dir(m.group(1))
        if not d or not os.path.isdir(d): return self._json(404, {'error':'олдсонгүй'})
        name = safe_name(urllib.parse.unquote(m.group(2)))
        n = int(self.headers.get('Content-Length', 0))
        if n <= 0 or n > 60_000_000: return self._json(400, {'error':'хэмжээ буруу'})
        dest = os.path.join(d, 'images', name)
        left = n
        with open(dest, 'wb') as f:
            while left > 0:
                chunk = self.rfile.read(min(1 << 20, left))
                if not chunk: break
                f.write(chunk); left -= len(chunk)
        return self._json(200, {'ok':True, 'name':name})

class S(socketserver.ThreadingTCPServer):
    allow_reuse_address = True; daemon_threads = True

if __name__ == '__main__':
    with S(('127.0.0.1', PORT), H) as srv:
        print(f'Фотограмметрийн сервер:  http://localhost:{PORT}')
        print(f'Ажлын хавтас: {JOBS}')
        srv.serve_forever()
