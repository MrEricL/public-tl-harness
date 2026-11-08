"""Start demo runs from the workbench and record what the harness does as it happens.

A run started here is the same run_folder() call as python demo.py: the same
profile, budgets and output files. The workbench adds an activity log
(activity.jsonl in the run folder) and a stop request that ends spending
through the demo's existing fatal-error path. API keys stay in memory.
"""
from __future__ import annotations

from datetime import datetime
import json
import math
import os
from pathlib import Path
import re
import shutil
import threading
import time

from .autonomous_provider import GenerationTransportError
from .demo import discover_sources, load_contract, load_replay, run_folder
from .demo_provider import DemoGenerator, PROVIDERS, provider_config

SAMPLE = Path(__file__).resolve().parents[1] / "samples" / "showcase"
SAMPLE_SOURCE, SAMPLE_GLOSSARY = SAMPLE / "source", SAMPLE / "terms" / "demo_glossary.json"
UPLOADS = ".workbench-uploads"
MAX_SOURCE_CHARS = 12000  # demo.py's default per-chapter ceiling


def key_env(config: dict) -> str:
    default_url, name = PROVIDERS[config["provider"]]
    return name if config["base_url"] == default_url.rstrip("/") else "TL_API_KEY"


def run_file(folder: Path, name: str) -> Path | None:
    """A file inside a run folder, or None if a symlink takes it outside that folder."""
    try:
        path = (folder / name).resolve()
        return path if folder.resolve() in path.parents else None
    except (OSError, RuntimeError):
        return None


def replayable(folder: Path) -> bool:
    try:
        config = json.loads(run_file(folder, "config.json").read_text(encoding="utf-8"))
    except (AttributeError, OSError, ValueError):
        return False
    return config.get("schema_version") == "portfolio-run.v1" and (folder / "cache").is_dir()


def _number(request: dict, name: str, kind=float):
    value = request.get(name)
    if value in (None, ""):
        return None
    try:
        number = kind(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name.replace('_', ' ').capitalize()} must be a number") from None
    if isinstance(number, float) and not math.isfinite(number):
        raise ValueError(f"{name.replace('_', ' ').capitalize()} must be a number")
    return number


def _file_name(name: object, suffixes: tuple[str, ...], what: str) -> str:
    base = Path(str(name or "")).name
    stem, suffix = Path(base).stem, Path(base).suffix.lower()
    if suffix not in suffixes:
        raise ValueError(f"{base or 'A file'}: {what} must be {' or '.join(suffixes)}")
    stem = re.sub(r"[^\w.-]+", "_", stem).strip("._") or "file"
    return stem + suffix


def save_uploads(folder: Path, chapters: list, glossary: dict | None) -> tuple[Path, Path | None]:
    """Write dropped files as demo.py expects them: one chapter per .txt file, the glossary beside the folder."""
    if not isinstance(chapters, list) or not chapters:
        raise ValueError("Add at least one chapter .txt file")
    source = folder / "chapters"
    source.mkdir(parents=True)
    seen = set()
    for item in chapters:
        name = _file_name(item.get("name") if isinstance(item, dict) else None, (".txt",), "chapters")
        if name.casefold() in seen:  # macOS volumes ignore case: Ch1.txt would overwrite ch1.txt
            raise ValueError(f"Two chapters are both named {name} (names that differ only in case count as the same)")
        seen.add(name.casefold())
        (source / name).write_text(str(item.get("text", "")), encoding="utf-8")
    glossary_path = None
    if glossary and not isinstance(glossary, dict):
        raise ValueError("Send the glossary as a file with a name and text")
    if glossary:
        glossary_path = folder / _file_name(glossary.get("name"), (".json", ".txt"), "the glossary")
        glossary_path.write_text(str(glossary.get("text", "")), encoding="utf-8")
    return source, glossary_path


def prepare(root: Path, request: dict) -> dict:
    """Validate a start request as demo.py validates its arguments, before any file or key is used."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = root / "workbench" / stamp
    suffix = 2
    while out.exists() or (root / UPLOADS / out.name).exists():
        out, suffix = root / "workbench" / f"{stamp}-{suffix}", suffix + 1
    kind = request.get("source")
    if kind == "replay":
        replay = (root / str(request.get("replay") or "")).resolve()
        if (replay != root and root not in replay.parents) or not replayable(replay):
            raise ValueError("Choose a recorded demo run that kept its cache folder")
        saved = load_replay(replay)
        return dict(out=out, source=replay / "inputs", glossary=replay / "glossary.json", config=saved["provider"],
                    compare=saved["compare"], limit=0, max_source_chars=saved["max_source_chars"],
                    replay=replay, api_key=None, mode="cache replay")
    provider = request.get("provider") or "openai"
    if provider not in PROVIDERS:
        raise ValueError("Choose one of the listed providers")
    # The page shows the endpoint field only for "custom"; a remembered URL must not
    # redirect a named provider's run, and the key pasted for it, somewhere else.
    base_url = str(request.get("base_url") or "").strip() or None
    if base_url and provider != "custom":
        raise ValueError("An endpoint URL is used only with the custom provider")
    thinking = request.get("thinking_mode") or ("disabled" if provider == "deepseek" else None)
    config = provider_config(provider, str(request.get("model") or "").strip(), base_url=base_url,
                             input_price=_number(request, "input_price"), output_price=_number(request, "output_price"),
                             max_usd=_number(request, "max_usd"), thinking_mode=thinking)
    limit = _number(request, "limit", int)
    limit = 3 if limit is None else limit
    api_key = str(request.get("api_key") or "").strip() or None
    if not api_key and not os.getenv(key_env(config)):
        raise ValueError(f"Paste an API key, or set {key_env(config)} before starting the workbench")
    if kind == "sample":
        source, glossary = SAMPLE_SOURCE, SAMPLE_GLOSSARY
        load_contract(glossary)
        discover_sources(source, out, limit, MAX_SOURCE_CHARS)
    elif kind == "upload":
        if not isinstance(request.get("chapters"), list) or not request["chapters"]:
            raise ValueError("Add at least one chapter .txt file")
        uploads = root / UPLOADS / out.name
        try:
            source, glossary = save_uploads(uploads, request["chapters"], request.get("glossary"))
            load_contract(glossary)
            discover_sources(source, out, limit, MAX_SOURCE_CHARS)
        except Exception:  # A refused start keeps no copy of the private chapters.
            shutil.rmtree(uploads, ignore_errors=True)
            raise
    else:
        raise ValueError("Choose chapter files, the sample story, or a recorded run to replay")
    return dict(out=out, source=source, glossary=glossary, config=config, compare=bool(request.get("compare", True)),
                limit=limit, max_source_chars=MAX_SOURCE_CHARS, replay=None, api_key=api_key, mode="live model")


class Activity:
    """Append-only events for one run, mirrored to activity.jsonl once run_folder has created the folder."""

    def __init__(self, out: Path):
        self.out, self.events, self.lock, self.started = out, [], threading.Lock(), time.monotonic()
        self.written = 0

    def emit(self, kind: str, **data) -> None:
        with self.lock:
            self.events.append(dict(seq=len(self.events), seconds=round(time.monotonic() - self.started, 2),
                                    kind=kind, **data))
            if self.out.is_dir():
                try:
                    with (self.out / "activity.jsonl").open("a", encoding="utf-8") as handle:
                        handle.writelines(json.dumps(event, ensure_ascii=False) + "\n"
                                          for event in self.events[self.written:])
                    self.written = len(self.events)
                except OSError:  # The page still has every event; the file catches up on the next one
                    pass         # (a batch that was partly written may then repeat a few lines).

    def since(self, after: int) -> list[dict]:
        with self.lock:
            return self.events[max(after, 0):]


def read_activity(folder: Path, after: int = 0) -> list[dict] | None:
    try:
        lines = run_file(folder, "activity.jsonl").read_text(encoding="utf-8").splitlines()
    except (AttributeError, OSError):
        return None
    return [json.loads(line) for line in lines[max(after, 0):] if line.strip()]


class Job:
    """One workbench run. Chapter progress is read from the files run_folder already writes."""

    def __init__(self, run_id: str, plan: dict):
        self.id, self.plan, self.out = run_id, plan, plan["out"]
        self.activity = Activity(self.out)
        self.status, self.stop_requested = "running", False
        self.calls, self.refused, self.started, self.finished = 0, 0, [], set()
        self.names: dict[str, str] = {}

    @property
    def current(self) -> str | None:
        return self.started[-1] if self.started else None

    def chapter_summary(self, chapter_id: str) -> dict:
        chapter = json.loads((self.out / "chapters" / chapter_id / "result.json").read_text(encoding="utf-8"))
        arms = chapter.get("arms") or {}
        events = (arms.get("harness") or {}).get("events") or []
        facts = 0
        try:
            for line in (self.out / "memory" / "story_memory" / "updates.jsonl").read_text(encoding="utf-8").splitlines():
                record = json.loads(line)
                facts += record.get("type") == "memory_entry" and record.get("chapter_id") == chapter_id
        except (OSError, ValueError):
            pass
        return dict(chapter=chapter_id, name=chapter.get("name"), status=chapter.get("status"),
                    error=chapter.get("error_type"), facts=facts,
                    kept=sum(e.get("kind") in {"patch_accepted", "term_alignment_applied"} for e in events),
                    rejected=sum(e.get("kind") == "patch_rejected" for e in events),
                    warnings=len((arms.get("harness") or {}).get("warnings") or []) + len(chapter.get("warnings") or []),
                    checks={arm: [sum(bool(c.get("passed")) for c in value.get("checks") or []), len(value.get("checks") or [])]
                            for arm, value in arms.items()})

    def sync(self, report: dict | None = None) -> None:
        """Emit chapter ends and starts that appeared on disk since the last model call."""
        for chapter_id in self.started:
            if chapter_id not in self.finished and (self.out / "chapters" / chapter_id / "result.json").is_file():
                self.finished.add(chapter_id)
                try:
                    summary = self.chapter_summary(chapter_id)
                except Exception as exc:  # Still close the chapter in the feed, without its counts.
                    self.activity.emit("activity_error", error=type(exc).__name__, message=str(exc))
                    summary = dict(chapter=chapter_id, name=self.names.get(chapter_id))
                self.activity.emit("chapter_finished", **summary)
        if not self.names and (self.out / "config.json").is_file():
            config = json.loads((self.out / "config.json").read_text(encoding="utf-8"))
            self.names = {item["id"]: item["name"] for item in config.get("inputs", [])}
        for path in sorted((self.out / "inputs").glob("*.txt")) if (self.out / "inputs").is_dir() else ():
            if path.stem not in self.started:
                self.started.append(path.stem)
                self.activity.emit("chapter_started", chapter=path.stem, name=self.names.get(path.stem, path.name),
                                   index=len(self.started), total=len(self.names) or None)
        for chapter in (report or {}).get("chapters", []):
            if chapter["id"] not in self.finished:
                self.finished.add(chapter["id"])
                self.activity.emit("chapter_finished", chapter=chapter["id"], name=chapter.get("name"),
                                   status=chapter.get("status"), error=chapter.get("error_type"))

    def safe_sync(self, report: dict | None = None) -> None:
        """The feed is bookkeeping: an error reading run files must not fail a model call or a finished run."""
        try:
            self.sync(report)
        except Exception as exc:
            self.activity.emit("activity_error", error=type(exc).__name__, message=str(exc))

    def run(self, client=None) -> None:
        plan, generator, status = self.plan, None, "failed"
        try:
            config = plan["config"]
            generator = TrackedGenerator(self, self.out, config, api_key=plan["api_key"], client=client,
                                         replay=plan["replay"] is not None,
                                         cache_read_dir=plan["replay"] / "cache" if plan["replay"] else None)
            self.activity.emit("run_started", mode=plan["mode"], provider=config["provider"], model=config["model"],
                               compare=plan["compare"])
            report = run_folder(plan["source"], self.out, config, glossary_path=plan["glossary"],
                                compare=plan["compare"], limit=plan["limit"], max_source_chars=plan["max_source_chars"],
                                generator=generator, replay_from=plan["replay"])
            # "Stopped" only if the request actually prevented a model call.
            status = "stopped" if self.refused else "done"  # The run is over; what follows is the feed.
            self.safe_sync(report)
            finished = dict(status=report.get("status"), stopped=bool(self.refused))
            try:
                scores = report.get("summary", {}).get("groups", {}).get("all", {})
                finished.update(resources=report.get("resources"), arms=report["arms"],
                                scores={arm: [value["passed"], value["total"]] for arm, value in scores.items()})
            except Exception as exc:
                self.activity.emit("activity_error", error=type(exc).__name__, message=str(exc))
            self.activity.emit("run_finished", **finished)
        except Exception as exc:  # The page shows the reason; keys never enter provider messages.
            self.activity.emit("run_failed", error=type(exc).__name__, message=str(exc))
        finally:
            try:
                if generator is not None:
                    generator.close()
            finally:  # Even if closing the client fails: drop the key and free the board for the next run.
                self.plan["api_key"] = None  # Finished jobs stay listed; the pasted key does not.
                self.status = status


class TrackedGenerator(DemoGenerator):
    """Reports each model call as it starts and ends; a stop request refuses the next request."""

    def __init__(self, job: Job, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.job = job

    def refuse_if_stopped(self) -> None:
        if self.job.stop_requested:
            self.job.refused += 1
            raise GenerationTransportError("Stopped from the workbench before the next model call")

    def _attempt(self, request: dict, *, attempt: int, correction_of: str | None = None) -> dict:
        if attempt > 1:  # DemoGenerator.call's schema-correction retry is a second paid request.
            self.refuse_if_stopped()
        return super()._attempt(request, attempt=attempt, correction_of=correction_of)

    def call(self, operation: str, prompt: str, schema: dict, **kwargs) -> dict:
        job = self.job
        job.safe_sync()
        self.refuse_if_stopped()
        job.calls += 1
        number, first, started, error = job.calls, len(self.receipts), time.monotonic(), None
        job.activity.emit("call_started", call=number, operation=operation, chapter=job.current)
        try:
            return super().call(operation, prompt, schema, **kwargs)
        except Exception as exc:
            error = type(exc).__name__
            raise
        finally:
            receipts = self.receipts[first:]

            def total(key):
                values = [receipt.get(key) for receipt in receipts]
                return sum(values) if values and all(value is not None for value in values) else None
            job.activity.emit("call_finished", call=number, operation=operation, chapter=job.current,
                              ok=error is None, error=error, attempts=len(receipts),
                              cached=bool(receipts) and all(r.get("cache_hit") for r in receipts),
                              input_tokens=total("input_tokens"), output_tokens=total("output_tokens"),
                              usd=total("estimated_usd"), duration=round(time.monotonic() - started, 2))


class JobBoard:
    """At most one run at a time, so budgets and the activity feed stay unambiguous."""

    def __init__(self, root: Path):
        # Re-entrant: start() holds the lock while it calls active().
        self.root, self.jobs, self.lock = root, {}, threading.RLock()

    def active(self) -> Job | None:
        with self.lock:  # start() may be adding a job from another request thread.
            return next((job for job in self.jobs.values() if job.status == "running"), None)

    def start(self, request: dict, *, client=None, background: bool = True) -> Job:
        with self.lock:
            if self.active():
                raise RuntimeError("A run is already in progress. Stop it or wait for it to finish.")
            plan = prepare(self.root, request)
            job = Job(plan["out"].relative_to(self.root).as_posix(), plan)
            self.jobs[job.id] = job
        if background:
            threading.Thread(target=job.run, kwargs=dict(client=client), daemon=True, name=f"workbench {job.id}").start()
        else:
            job.run(client)
        return job

    def stop(self) -> Job | None:
        job = self.active()
        if job:
            job.stop_requested = True
        return job
