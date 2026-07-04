import json, time, urllib.request

BASE = "http://localhost:8020"

def post(path, payload):
    req = urllib.request.Request(
        BASE + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=60)

# 1) agent
r = post("/api/agent", {"text": "قولي جملة ترحيب قصيرة", "model": "gpt-4o-mini"})
d = json.load(r)
reply = d["reply"]
print("AGENT reply:", reply, "| model:", d["model"])

# 2) tts stream that reply
t0 = time.time()
r = post("/api/tts/stream", {"text": reply, "language": "ar"})
data = r.read()
dur = len(data) / 2 / 24000
print(f"TTS-STREAM: {len(data)} bytes = {dur:.2f}s audio in {time.time()-t0:.2f}s  (HTTP {r.status})")
print("OK" if len(data) > 10000 else "FAIL: too little audio")
