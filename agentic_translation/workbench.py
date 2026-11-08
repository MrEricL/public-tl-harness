"""Local workbench for the demo harness: python -m agentic_translation.workbench.

Start a run from dropped chapter files (or the sample story, or a replay),
watch each model call as the harness makes it, then inspect what changed.
Runs go through demo.run_folder, so they write the same files as demo.py.
The server binds to localhost and loads no third-party script or font: run
folders hold private chapter text. API keys pasted into the page stay in
this process's memory and are never written.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import difflib
from functools import partial
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
from urllib.parse import parse_qs, unquote, urlsplit
import webbrowser

from .demo_provider import PROVIDERS
from .workbench_runs import SAMPLE_GLOSSARY, SAMPLE_SOURCE, JobBoard, read_activity, replayable, run_file

STATIC = Path(__file__).with_name("static")
PAGES = {"/": "workbench.html", "/workbench.css": "workbench.css", "/workbench.js": "workbench.js"}
SKIP_DIRS = {"cache", "calls", "chapters", "delivery", "inputs", "memory", "sessions", "cases", "tasks", "records"}
SERVED_TYPES = {".txt": "text/plain", ".md": "text/plain", ".json": "application/json", ".jsonl": "text/plain",
                ".html": "text/html", ".epub": "application/epub+zip", ".svg": "image/svg+xml", ".png": "image/png"}
TOKEN = re.compile(r"\s+|[\w’'-]+|[^\w\s]", re.UNICODE)
MAX_BODY = 20 * 1024 * 1024


def _read_json(path: Path | None):
    """None for a missing, unreadable or out-of-folder (run_file gave None) file."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (AttributeError, OSError, ValueError):
        return None


def _is_demo_results(value) -> bool:
    """demo.py and the repair bench write arms plus per-chapter arm records."""
    if not isinstance(value, dict):
        return False
    chapters = value.get("chapters")
    return (isinstance(value.get("arms"), list) and isinstance(chapters, list)
            and all(isinstance(c, dict) and isinstance(c.get("arms"), dict) for c in chapters))


def _modified(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(timespec="seconds")


def discover(root: Path, max_depth: int = 3) -> dict:
    """Demo runs (results.json) and other harness reports (report.html), newest first."""
    runs, reports = [], []

    def walk(folder: Path, depth: int):
        try:
            children = sorted(os.scandir(folder), key=lambda e: e.name)
        except OSError:
            return
        names = {entry.name for entry in children}
        rel = folder.relative_to(root).as_posix()
        if "results.json" in names and _is_demo_results(results := _read_json(run_file(folder, "results.json"))):
            try:
                runs.append(summarize_run(folder, rel, results))
            except (AttributeError, IndexError, KeyError, TypeError):
                pass  # One malformed results.json hides its own folder, not the whole list.
            else:
                if depth or "workbench" not in names:
                    return
                # The served folder is itself a run: still list the runs the workbench started inside it.
                children = [entry for entry in children if entry.name == "workbench"]
        if "report.html" in names:
            reports.append(dict(id=rel, name=folder.name if rel == "." else rel, modified=_modified(folder / "report.html")))
        if depth < max_depth:
            for entry in children:
                if entry.is_dir(follow_symlinks=False) and entry.name not in SKIP_DIRS and not entry.name.startswith("."):
                    walk(Path(entry.path), depth + 1)

    walk(root, 0)
    runs.sort(key=lambda row: row["modified"], reverse=True)
    reports.sort(key=lambda row: row["modified"], reverse=True)
    return dict(root=str(root), runs=runs, reports=reports)


def summarize_run(folder: Path, rel: str, results: dict) -> dict:
    config = _read_json(run_file(folder, "config.json")) or {}
    provider = config.get("provider") if isinstance(config.get("provider"), dict) else {}
    arms = results["arms"]
    scores = (results.get("summary") or {}).get("groups", {}).get("all", {})
    return dict(id=rel, name=folder.name if rel == "." else rel, status=results.get("status"), mode=results.get("mode"), arms=arms,
                chapters=len(results["chapters"]), model=provider.get("model"), provider=provider.get("provider"),
                first=scores.get(arms[0]) if arms else None, last=scores.get(arms[-1]) if arms else None,
                resources=results.get("resources"), modified=_modified(folder / "results.json"),
                replayable=replayable(folder), activity=(folder / "activity.jsonl").is_file())


def memory_records(folder: Path) -> list[dict]:
    """Story-memory updates in the order the run accepted them; absent for runs without memory."""
    path = run_file(folder, "memory/story_memory/updates.jsonl")
    records = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (AttributeError, OSError):
        return records
    for line in lines:
        try:
            records.append(json.loads(line))
        except ValueError:
            continue
    return records


def load_run(root: Path, run_id: str) -> dict:
    folder = resolve(root, run_id)
    results = _read_json(run_file(folder, "results.json"))
    if not _is_demo_results(results):
        raise FileNotFoundError(run_id)
    config = _read_json(run_file(folder, "config.json")) or {}
    files = [path for path in ("delivery/book.txt", "delivery/book.epub", "report.html", "report.md", "results.json")
             if (folder / path).is_file()]
    return dict(id=run_id, summary=summarize_run(folder, run_id, results), results=results, config=config,
                glossary=_read_json(run_file(folder, "glossary.json")) or config.get("glossary") or [],
                memory=memory_records(folder), files=files)


def setup(root: Path, board: JobBoard) -> dict:
    """What the new-run form needs: providers, whether each key is already set (never its value), and replays."""
    sample = _read_json(SAMPLE_GLOSSARY) or []
    active = board.active()
    return dict(
        providers={name: dict(base_url=url, key_env=env, key_set=bool(os.getenv(env))) for name, (url, env) in PROVIDERS.items()},
        sample=dict(chapters=len(list(SAMPLE_SOURCE.glob("*.txt"))), terms=len(sample)),
        replays=[dict(id=run["id"], name=run["name"], chapters=run["chapters"], model=run["model"], status=run["status"])
                 for run in discover(root)["runs"] if run["replayable"]],
        active=active and dict(id=active.id, status=active.status))


def activity(root: Path, board: JobBoard, run_id: str, after: int) -> dict:
    job = board.jobs.get(run_id)
    if job:
        return dict(id=run_id, status=job.status, stopping=job.stop_requested, events=job.activity.since(after))
    folder = resolve(root, run_id)
    events = read_activity(folder, after)
    if events is None:
        raise FileNotFoundError(run_id)
    # No live job: the run ended, or its process died. Only a run whose feed reached
    # run_finished/run_failed, or that wrote results.json, counts as recorded.
    every = read_activity(folder, 0) or []
    ended = any(e.get("kind") in {"run_finished", "run_failed"} for e in every) or (folder / "results.json").is_file()
    return dict(id=run_id, status="recorded" if ended else "interrupted", stopping=False, events=events)


def _paragraphs(text: str) -> list[str]:
    """Non-empty lines; the page aligns Chinese and English paragraphs by the same rule."""
    return [line.strip() for line in (text or "").splitlines() if line.strip()]


def _ops(before: str, after: str) -> list[list[str]]:
    a, b = TOKEN.findall(before), TOKEN.findall(after)
    ops = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag == "equal":
            ops.append(["equal", "".join(a[i1:i2])])
            continue
        if i2 > i1:
            ops.append(["delete", "".join(a[i1:i2])])
        if j2 > j1:
            ops.append(["insert", "".join(b[j1:j2])])
    return ops


def word_diff(before: str, after: str) -> dict:
    """Word-level edits, per paragraph when both texts keep the same paragraph count."""
    old, new = _paragraphs(before), _paragraphs(after)
    if len(old) == len(new):
        paragraphs = [_ops(a, b) for a, b in zip(old, new)]
        return dict(aligned=True, paragraphs=paragraphs, changed=sum(any(op[0] != "equal" for op in p) for p in paragraphs))
    ops = _ops("\n\n".join(old), "\n\n".join(new))
    return dict(aligned=False, paragraphs=[ops], changed=int(any(op[0] != "equal" for op in ops)))


def resolve(root: Path, relative: str) -> Path:
    """Refuse anything outside the served runs folder, including through symlinks."""
    target = (root / relative).resolve()
    if target != root and root not in target.parents:
        raise PermissionError(relative)
    return target


class Handler(BaseHTTPRequestHandler):
    server_version = "TranslationWorkbench/1"

    def __init__(self, *args, root: Path, board: JobBoard, **kwargs):
        self.root, self.board = root, board
        super().__init__(*args, **kwargs)

    def log_message(self, format, *args):  # Quiet by default; errors still reach the browser.
        pass

    def send(self, status: int, body: bytes, content_type: str, sandbox: bool = False):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        if sandbox:
            # Run files (saved reports, SVGs) can hold chapter or model text. Sandboxed, they get no
            # scripts and an opaque origin, so nothing in them can read runs or start one.
            self.send_header("Content-Security-Policy", "sandbox allow-downloads allow-popups")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, value, status: int = 200):
        self.send(status, json.dumps(value, ensure_ascii=False).encode(), "application/json; charset=utf-8")

    def local(self) -> bool:
        # A page on another site that rebinds its hostname to 127.0.0.1 must not read private chapters.
        if self.headers.get("Host", "").rsplit(":", 1)[0] in {"127.0.0.1", "localhost"}:
            return True
        self.send_json(dict(error="Open the workbench at 127.0.0.1"), HTTPStatus.FORBIDDEN)
        return False

    def same_origin(self) -> bool:
        try:  # A malformed Origin (bad port, broken IPv6 literal) is refused, not a crash.
            origin = urlsplit(self.headers.get("Origin", ""))
            return origin.hostname in {"127.0.0.1", "localhost"} and origin.port == self.server.server_address[1]
        except ValueError:
            return False

    def do_GET(self):
        if not self.local():
            return
        url = urlsplit(self.path)
        query = {key: values[-1] for key, values in parse_qs(url.query).items()}
        try:
            if url.path in PAGES:
                path = STATIC / PAGES[url.path]
                kind = {".html": "text/html", ".css": "text/css", ".js": "text/javascript"}[path.suffix]
                return self.send(200, path.read_bytes(), f"{kind}; charset=utf-8")
            if url.path == "/api/runs":
                return self.send_json(discover(self.root))
            if url.path == "/api/setup":
                return self.send_json(setup(self.root, self.board))
            if url.path == "/api/run":
                return self.send_json(load_run(self.root, query.get("id", "")))
            if url.path == "/api/activity":
                return self.send_json(activity(self.root, self.board, query.get("id", ""), int(query.get("after") or 0)))
            if url.path == "/api/diff":
                results = _read_json(run_file(resolve(self.root, query.get("id", "")), "results.json"))
                chapter = next(c for c in results["chapters"] if c.get("id") == query.get("chapter"))
                texts = chapter.get("texts") or {}
                return self.send_json(word_diff(texts.get(query.get("a"), ""), texts.get(query.get("b"), "")))
            if url.path.startswith("/runs/"):
                # Path-style URLs keep the relative links inside saved HTML reports working.
                path = resolve(self.root, unquote(url.path[len("/runs/"):]))
                kind = SERVED_TYPES.get(path.suffix.lower())
                if kind is None or not path.is_file():
                    raise FileNotFoundError(url.path)
                charset = "; charset=utf-8" if kind.startswith("text/") or kind.endswith("json") else ""
                return self.send(200, path.read_bytes(), kind + charset, sandbox=True)
            self.send_json(dict(error="Not found"), HTTPStatus.NOT_FOUND)
        except PermissionError:
            self.send_json(dict(error="Path is outside the runs folder"), HTTPStatus.FORBIDDEN)
        except (FileNotFoundError, StopIteration, TypeError, KeyError, ValueError):
            self.send_json(dict(error="No such run, chapter or file"), HTTPStatus.NOT_FOUND)

    def do_POST(self):
        if not self.local():
            return
        # Only this page may start or stop a run: other origins fail here, or at the JSON preflight.
        if not self.same_origin():
            return self.send_json(dict(error="Start runs from the workbench page"), HTTPStatus.FORBIDDEN)
        if not self.headers.get("Content-Type", "").startswith("application/json"):
            return self.send_json(dict(error="Send JSON"), HTTPStatus.UNSUPPORTED_MEDIA_TYPE)
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = -1
        if length < 0:  # A negative length would make the read wait for the client to hang up.
            return self.send_json(dict(error="Send a valid Content-Length"), HTTPStatus.BAD_REQUEST)
        if length > MAX_BODY:
            return self.send_json(dict(error="Those files are too large for one run; send fewer chapters"),
                                  HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
            if not isinstance(body, dict):
                raise ValueError("Send a JSON object")
            if self.path == "/api/start":
                job = self.board.start(body)
                return self.send_json(dict(id=job.id))
            if self.path == "/api/stop":
                job = self.board.stop()
                return self.send_json(dict(id=job and job.id))
            self.send_json(dict(error="Not found"), HTTPStatus.NOT_FOUND)
        except RuntimeError as exc:
            self.send_json(dict(error=str(exc)), HTTPStatus.CONFLICT)
        except (ValueError, OSError) as exc:
            self.send_json(dict(error=str(exc)), HTTPStatus.BAD_REQUEST)
        except (AttributeError, KeyError, TypeError) as exc:  # a field of the wrong type or a missing key
            self.send_json(dict(error=f"Malformed request or recorded run ({type(exc).__name__})"),
                           HTTPStatus.BAD_REQUEST)


def serve(root: Path, host: str = "127.0.0.1", port: int = 8765) -> ThreadingHTTPServer:
    root = root.resolve()
    board = JobBoard(root)
    server = ThreadingHTTPServer((host, port), partial(Handler, root=root, board=board))
    server.board = board
    return server


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Start, watch and inspect demo runs in a local page.")
    parser.add_argument("runs", nargs="?", type=Path, default=Path("runs"), help="Runs folder (default: runs)")
    parser.add_argument("--port", type=int, default=8765, help="Local port; 0 picks a free one")
    parser.add_argument("--open", action="store_true", help="Open the page in your browser")
    args = parser.parse_args(argv)
    if not args.runs.is_dir():
        print(f"Workbench input error: {args.runs} is not a folder. Create it, "
              "or pass the folder that holds your runs.")
        return 2
    server = serve(args.runs, port=args.port)
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    print(f"Translation workbench: {url} (runs folder {args.runs.resolve()}; Ctrl+C to stop)", flush=True)
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
