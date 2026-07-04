#!/usr/bin/env python3
"""
Tiny HTTP proxy to work around an ISP (TE Data / Telecom Egypt) middlebox that
hijacks plain-HTTP (port 80) traffic and injects a redirect to
`megaplusredirection.tedata.net`.

ngrok's agent fetches its certificate revocation list over plain HTTP
(`http://crl.ngrok-agent.com/ngrok.crl`). The middlebox replaces that binary
CRL with an HTML redirect, so ngrok fails auth with:
    "failed to fetch CRL: asn1: structure error: length too large"

This proxy, when ngrok routes through it (http_proxy / https_proxy env), does two
things:
  * do_GET  — for any intercepted http:// CRL request, re-fetches the SAME url
              over HTTPS (which the middlebox does NOT touch) and returns the
              real DER bytes.  Other http GETs are proxied normally over IPv4.
  * do_CONNECT — tunnels https traffic, force-dialing IPv4 (this host's IPv6
              egress is broken, so IPv6-first DNS results must be skipped).

Listens on 127.0.0.1:8899.
"""
import http.server
import select
import socket
import ssl
import sys
from urllib.parse import urlsplit

LISTEN = ("127.0.0.1", 8899)


def dial_ipv4(host: str, port: int, timeout: float = 15.0) -> socket.socket:
    """Connect to host:port using IPv4 only (IPv6 egress is broken here)."""
    last = None
    for fam, stype, proto, _, addr in socket.getaddrinfo(
        host, port, socket.AF_INET, socket.SOCK_STREAM
    ):
        try:
            s = socket.socket(fam, stype, proto)
            s.settimeout(timeout)
            s.connect(addr)
            s.settimeout(None)
            return s
        except OSError as e:
            last = e
    raise OSError(f"IPv4 connect failed for {host}:{port}: {last}")


def http_get(url: str, timeout: float = 15.0):
    """GET a url over IPv4; auto-upgrade the ngrok CRL to HTTPS. Returns
    (status, content_type, body_bytes). Assumes a small, non-chunked body."""
    u = urlsplit(url)
    https = u.scheme == "https"
    host = u.hostname
    port = u.port or (443 if https else 80)
    path = (u.path or "/") + (("?" + u.query) if u.query else "")

    sock = dial_ipv4(host, port, timeout)
    if https:
        ctx = ssl.create_default_context()
        sock = ctx.wrap_socket(sock, server_hostname=host)

    req = (
        f"GET {path} HTTP/1.1\r\nHost: {host}\r\n"
        "User-Agent: crl-proxy\r\nAccept: */*\r\nConnection: close\r\n\r\n"
    )
    sock.sendall(req.encode())
    buf = b""
    while True:
        try:
            chunk = sock.recv(65536)
        except OSError:
            break
        if not chunk:
            break
        buf += chunk
    try:
        sock.close()
    except OSError:
        pass

    head, _, body = buf.partition(b"\r\n\r\n")
    lines = head.split(b"\r\n")
    status = int(lines[0].split()[1]) if lines and len(lines[0].split()) > 1 else 502
    ctype = "application/octet-stream"
    chunked = False
    for line in lines[1:]:
        low = line.lower()
        if low.startswith(b"content-type:"):
            ctype = line.split(b":", 1)[1].strip().decode(errors="replace")
        elif low.startswith(b"transfer-encoding:") and b"chunked" in low:
            chunked = True
    if chunked:
        body = _dechunk(body)
    return status, ctype, body


def _dechunk(data: bytes) -> bytes:
    out = b""
    while data:
        size_line, _, rest = data.partition(b"\r\n")
        try:
            n = int(size_line.strip(), 16)
        except ValueError:
            break
        if n == 0:
            break
        out += rest[:n]
        data = rest[n + 2:]
    return out


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):  # silence
        pass

    def do_CONNECT(self):
        print(f"[proxy] CONNECT {self.path}", flush=True)
        host, _, port = self.path.partition(":")
        try:
            upstream = dial_ipv4(host, int(port or 443))
        except OSError as e:
            self.send_error(502, str(e))
            return
        self.send_response(200, "Connection established")
        self.end_headers()
        self._tunnel(self.connection, upstream)

    def _tunnel(self, a: socket.socket, b: socket.socket):
        socks = [a, b]
        try:
            while True:
                r, _, x = select.select(socks, [], socks, 120)
                if x or not r:
                    break
                stop = False
                for s in r:
                    other = b if s is a else a
                    try:
                        data = s.recv(65536)
                    except OSError:
                        stop = True
                        break
                    if not data:
                        stop = True
                        break
                    try:
                        other.sendall(data)
                    except OSError:
                        stop = True
                        break
                if stop:
                    break
        finally:
            for s in (a, b):
                try:
                    s.close()
                except OSError:
                    pass

    def do_GET(self):
        print(f"[proxy] GET {self.path}", flush=True)
        url = self.path  # absolute-form when proxied
        # The middlebox only mangles plain HTTP. Re-fetch the CRL over HTTPS.
        if url.startswith("http://") and "crl.ngrok-agent.com" in url:
            url = "https://" + url[len("http://"):]
        try:
            status, ctype, body = http_get(url)
        except Exception as e:  # noqa: BLE001
            self.send_error(502, f"proxy fetch failed: {e}")
            return
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        try:
            self.wfile.write(body)
        except OSError:
            pass


class ThreadingHTTPServer(http.server.ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


if __name__ == "__main__":
    srv = ThreadingHTTPServer(LISTEN, Handler)
    print(f"CRL-fix proxy listening on http://{LISTEN[0]}:{LISTEN[1]}", flush=True)
    srv.serve_forever()
