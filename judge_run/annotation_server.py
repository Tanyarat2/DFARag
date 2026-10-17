#!/usr/bin/env python3
"""Serve a local run picker and the latest annotation view for each judge run.

No database or API key is used by this server. Judge results are read-only;
human annotations are saved separately in each run's annotations/PS.json,
annotations/KH.json, and annotations/WN.json files.
"""

import argparse
import html
import json
import pathlib
import re
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlparse

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import make_annotation as annotation  # noqa: E402
from annotation_store import AnnotationStore  # noqa: E402

NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z")
JUDGES = ("mtrag",)


def available_runs(runs_dir):
    if not runs_dir.is_dir():
        return []
    rows = []
    for folder in sorted(runs_dir.iterdir(), key=lambda p: p.name.lower()):
        if not folder.is_dir() or not NAME_PATTERN.fullmatch(folder.name):
            continue
        input_exists = (folder / "input.json").is_file()
        found = [name for name in JUDGES if (folder / ("out.%s.json" % name)).is_file()]
        if input_exists or found:
            rows.append({"name": folder.name, "has_input": input_exists, "judges": found})
    return rows


def picker_page(runs_dir, port):
    rows = available_runs(runs_dir)
    if rows:
        content = []
        for row in rows:
            name = html.escape(row["name"])
            statuses = " ".join("<span class='tag'>%s</span>" % html.escape(n) for n in row["judges"])
            if not statuses:
                statuses = "<span class='muted'>no judge output yet</span>"
            elif len(row["judges"]) < len(JUDGES):
                missing = [n for n in JUDGES if n not in row["judges"]]
                statuses += " " + " ".join("<span class='missing'>%s: not run</span>" % html.escape(n)
                                               for n in missing)
            content.append("<a class='run' href='/run/%s'><strong>%s</strong>"
                           "<span>%s</span></a>" % (name, name, statuses))
        listing = "".join(content)
    else:
        listing = "<p class='muted'>No runs found yet. Create one with <code>python run.py prepare --name pilot7_clapnq</code>.</p>"
    return """<!doctype html><html lang='en'><head><meta charset='utf-8'>
<meta name='viewport' content='width=device-width, initial-scale=1'><title>RAG judge runs</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#f4f5f7;color:#1b1f23;font:15px system-ui,sans-serif}
main{max-width:900px;margin:36px auto;padding:0 18px}h1{font-size:25px;margin:0 0 8px}
p{line-height:1.55}.muted{color:#68717d}.runs{display:grid;gap:9px;margin-top:24px}
a.run{display:grid;gap:8px;padding:14px;background:white;border:1px solid #d5d8de;border-radius:6px;color:inherit;text-decoration:none}
a.run:hover{border-color:#27827c;box-shadow:0 3px 12px #1b1f2314}.run strong{font-size:16px}
.tag,.missing{display:inline-block;margin:0 5px 0 0;padding:3px 7px;border-radius:4px;background:#d9eee3;color:#205b38;font:12px ui-monospace,monospace}
.missing{background:#eceef1;color:#59616b}code{background:#e9ebee;padding:2px 4px}
.actions{display:flex;gap:12px;align-items:center;margin-top:20px}button{padding:8px 12px;border:1px solid #b9c0c8;border-radius:4px;background:white;cursor:pointer}
</style></head><body><main><h1>RAG judge runs</h1>
<p class='muted'>Local run folders; no database or browser API key. The list refreshes automatically while this page is open. Select a run to annotate the judge outputs currently available.</p>
<div class='actions'><button onclick='location.reload()'>Refresh list</button><span class='muted'>This page checks for new runs every 10 seconds.</span></div>
<section class='runs'>%s</section>
</main><script>setTimeout(()=>location.reload(),10000)</script></body></html>""" % listing


def make_handler(runs_dir):
    runs_dir = pathlib.Path(runs_dir)
    store = AnnotationStore(runs_dir)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = unquote(urlparse(self.path).path)
            api_match = re.fullmatch(r"/api/annotations/([^/]+)/([^/]+)", path)
            if api_match:
                try:
                    data = store.read(*api_match.groups())
                except FileNotFoundError:
                    self.respond_json(404, {"error": "Run not found."})
                except ValueError as exc:
                    self.respond_json(400, {"error": str(exc)})
                except (OSError, RuntimeError) as exc:
                    self.respond_json(500, {"error": str(exc)})
                else:
                    self.respond_json(200, data)
                return
            if path == "/" or path == "/index.html":
                self.respond(200, picker_page(runs_dir, self.server.server_port))
                return
            match = re.fullmatch(r"/run/([^/]+)", path)
            if not match or not NAME_PATTERN.fullmatch(match.group(1)):
                self.respond(404, "Not found")
                return
            work_dir = runs_dir / match.group(1)
            if not work_dir.is_dir():
                self.respond(404, "Run not found")
                return
            try:
                data = annotation.build_data(work_dir)
                data["annotation_api"] = "/api/annotations/" + match.group(1)
                page = annotation.render_page(data)
            except SystemExit as exc:
                message = html.escape(str(exc))
                page = ("<!doctype html><meta charset='utf-8'><title>Run unavailable</title>"
                        "<p>%s</p><p><a href='/'>Back to runs</a></p>" % message)
            else:
                nav = ("<nav aria-label='Run navigation' style='padding:8px 24px;"
                       "background:#fff;border-bottom:1px solid #dce3e1;font:12px system-ui'>"
                       "<a href='/'>All runs</a> &nbsp; / &nbsp; "
                       "<a href='/run/%s'>Refresh latest judge outputs</a></nav>" %
                       html.escape(match.group(1), quote=True))
                page = page.replace("<body>", "<body>" + nav, 1)
            self.respond(200, page)

        def do_POST(self):
            path = unquote(urlparse(self.path).path)
            match = re.fullmatch(r"/api/annotations/([^/]+)/([^/]+)", path)
            if not match:
                self.respond_json(404, {"error": "Not found."})
                return
            origin = self.headers.get("Origin")
            if origin and origin not in ("http://" + self.headers.get("Host", ""),
                                         "https://" + self.headers.get("Host", "")):
                self.respond_json(403, {"error": "Cross-origin annotation writes are not allowed."})
                return
            if self.headers.get("Content-Type", "").split(";", 1)[0].strip() != "application/json":
                self.respond_json(415, {"error": "Use application/json."})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length < 1 or length > 4 * 1024 * 1024:
                    self.respond_json(413, {"error": "Annotation request must be between 1 byte and 4 MiB."})
                    return
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict) or set(payload) != {"changes"}:
                    raise ValueError("Send a changes object.")
                run, annotator_name = match.groups()
                # Validate the destination before reading inputs or creating files.
                store.path(run, annotator_name)
                valid_keys = {row["key"] for row in annotation.build_data(runs_dir / run)["rows"]}
                store.patch(run, annotator_name, payload["changes"], valid_keys)
            except FileNotFoundError:
                self.respond_json(404, {"error": "Run not found."})
            except (ValueError, UnicodeError) as exc:
                self.respond_json(400, {"error": str(exc)})
            except (OSError, RuntimeError, SystemExit) as exc:
                self.respond_json(500, {"error": str(exc)})
            else:
                self.respond_json(200, {"saved": True, "annotator": annotator_name})

        def respond_json(self, status, data):
            self.respond(status, json.dumps(data, ensure_ascii=False), "application/json; charset=utf-8")

        def respond(self, status, body, content_type="text/html; charset=utf-8"):
            payload = body.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, fmt, *args):
            print("[%s] %s" % (self.log_date_time_string(), fmt % args))

    return Handler


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--runs-dir", type=pathlib.Path, default=HERE / "runs")
    parser.add_argument("--host", default="127.0.0.1", help="local-only by default")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(args.runs_dir.resolve()))
    url = "http://%s:%d/" % (args.host, server.server_port)
    print("RAG judge annotation: %s" % url, flush=True)
    print("Press Ctrl+C to stop. Judge runs can continue in another PowerShell window.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping annotation server.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
