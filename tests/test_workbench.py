"""Tests for the local, read-only run browser in agentic_translation/workbench.py.

The demo run fixture uses the same scripted-transport `Client` pattern as
test_portfolio_demo.py: a fake OpenAI-shaped client that returns canned JSON.
It exercises the workbench's file plumbing and HTTP routes, not model quality.
"""
from __future__ import annotations

import http.client
import json
import os
import sys
import threading
import time
from types import SimpleNamespace as NS
import urllib.error
import urllib.request

import pytest

from agentic_translation.demo import run_folder
from agentic_translation.demo_provider import DemoGenerator, provider_config
from agentic_translation import workbench as wb
from agentic_translation import workbench_runs as wr


class Client:
    """Minimal scripted transport fixture, mirroring test_portfolio_demo.Client."""
    def __init__(self):
        self.chat = NS(completions=self)

    def create(self, **kw):
        user = kw['messages'][1]['content']
        payload = json.JSONDecoder().raw_decode(user[user.index('\n{')+1:])[0]
        if user.startswith('Extract'):
            raw = json.dumps(dict(entries=[], terms=[]))
        elif user.startswith('Translate'):
            raw = '\n\n'.join(f'<<<SEGMENT:{row["segment_id"]}>>>\nLin Qing dodged the arrow.'
                               for row in payload['source_segments'])
        else:
            raw = json.dumps(dict(summary='No material issue', findings=[], patches=[]))
        return NS(model='test-model', usage=NS(prompt_tokens=10, completion_tokens=5),
                   choices=[NS(finish_reason='stop', message=NS(content=raw))])


@pytest.fixture
def root(tmp_path):
    """A runs root with one real demo run, one bare-report folder, one non-run folder."""
    source = tmp_path / 'source'; source.mkdir()
    (source / 'chapter2.txt').write_text('林青躲开了箭。', encoding='utf-8')
    out = tmp_path / 'run1'
    config = provider_config('openai', 'test-model')
    gen = DemoGenerator(out, config, client=Client())
    gen.evidence_mode = 'scripted integration fixture'
    report = run_folder(source, out, config, compare=True, generator=gen)
    assert report['status'] == 'delivered'
    (tmp_path / 'old-report').mkdir()
    (tmp_path / 'old-report' / 'report.html').write_text('<html>old</html>', encoding='utf-8')
    (tmp_path / 'weird').mkdir()
    (tmp_path / 'weird' / 'results.json').write_text(json.dumps({'assignments': []}), encoding='utf-8')
    return tmp_path, report


def test_discover_finds_run_and_report_and_ignores_other_schema(root):
    tmp_path, report = root
    found = wb.discover(tmp_path)
    run = next(r for r in found['runs'] if r['id'] == 'run1')
    assert run['status'] == 'delivered'
    assert run['arms'] == report['arms']
    assert run['chapters'] == len(report['chapters'])
    assert {r['id'] for r in found['reports']} == {'old-report'}
    assert all(r['id'] != 'weird' for r in found['runs'] + found['reports'])


def test_load_run_returns_files_and_rejects_bad_ids(root):
    tmp_path, report = root
    loaded = wb.load_run(tmp_path, 'run1')
    assert loaded['results']['status'] == 'delivered'
    assert loaded['glossary'] == []
    assert 'delivery/book.txt' in loaded['files']
    with pytest.raises(FileNotFoundError):
        wb.load_run(tmp_path, 'weird')
    with pytest.raises(PermissionError):
        wb.load_run(tmp_path, '../outside')


def test_word_diff_aligned_unaligned_and_identical():
    # Alignment is by non-empty-line count; blank lines between them don't count as paragraphs.
    before = 'one two three\n\nfour five six'
    after = 'one deux three\n\nfour five six'
    aligned = wb.word_diff(before, after)
    assert aligned['aligned'] is True
    assert len(aligned['paragraphs']) == 2
    tags = [op[0] for op in aligned['paragraphs'][0]]
    assert 'delete' in tags and 'insert' in tags
    assert all(op[0] == 'equal' for op in aligned['paragraphs'][1])
    assert aligned['changed'] == 1

    unaligned = wb.word_diff('a\nb', 'just one line')
    assert unaligned['aligned'] is False

    same = wb.word_diff('identical text', 'identical text')
    assert same['changed'] == 0


def test_http_routes_and_security(root):
    tmp_path, report = root
    server = wb.serve(tmp_path, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f'http://127.0.0.1:{server.server_address[1]}'
    try:
        with urllib.request.urlopen(f'{base}/') as resp:
            assert resp.status == 200 and 'text/html' in resp.headers['Content-Type']

        with urllib.request.urlopen(f'{base}/api/runs') as resp:
            runs = json.loads(resp.read())
            assert any(r['id'] == 'run1' for r in runs['runs'])

        with urllib.request.urlopen(f'{base}/api/run?id=run1') as resp:
            loaded = json.loads(resp.read())
            assert loaded['id'] == 'run1'

        chapter_id = report['chapters'][0]['id']
        with urllib.request.urlopen(
                f'{base}/api/diff?id=run1&chapter={chapter_id}&a=contextual&b=harness') as resp:
            diff = json.loads(resp.read())
            assert 'aligned' in diff and 'changed' in diff

        with urllib.request.urlopen(f'{base}/runs/run1/delivery/book.txt') as resp:
            assert resp.status == 200
        (tmp_path / 'run1' / 'evil.html').write_text('<script>fetch("/api/runs")</script>', encoding='utf-8')
        with urllib.request.urlopen(f'{base}/runs/run1/evil.html') as resp:
            # Run files may hold chapter or model text: no scripts, opaque origin (no allow-scripts,
            # no allow-same-origin), so nothing in them can call the API.
            assert 'text/html' in resp.headers['Content-Type']
            assert resp.headers['Content-Security-Policy'] == 'sandbox allow-downloads allow-popups'
        with urllib.request.urlopen(f'{base}/') as resp:
            assert 'Content-Security-Policy' not in resp.headers

        for attempt_path in ('/runs/%2e%2e/pyproject.toml', '/runs/..%2fpyproject.toml'):
            try:
                resp = urllib.request.urlopen(f'{base}{attempt_path}')
                assert resp.status != 200
            except urllib.error.HTTPError as err:
                assert err.code in (403, 404)

        req = urllib.request.Request(f'{base}/api/runs', headers={'Host': 'evil.example'})
        try:
            urllib.request.urlopen(req)
            assert False, 'expected 403 for a spoofed Host header'
        except urllib.error.HTTPError as err:
            assert err.code == 403
    finally:
        server.shutdown()
        server.server_close()


def test_main_reports_missing_runs_folder(tmp_path):
    assert wb.main([str(tmp_path / 'missing')]) == 2


def _no_secret_on_disk(folder, secret):
    """Walk every file under folder and confirm the secret byte string never appears."""
    needle = secret.encode()
    for path in folder.rglob('*'):
        if path.is_file():
            assert needle not in path.read_bytes(), f'{secret!r} leaked into {path}'


def test_jobboard_start_sample_run_records_activity_and_never_persists_key(tmp_path):
    board = wr.JobBoard(tmp_path)
    job = board.start(dict(source='sample', provider='openai', model='test-model', api_key='sk-test',
                           compare=True, limit=2), client=Client(), background=False)
    assert job.status == 'done'

    events = wr.read_activity(tmp_path / job.id)
    kinds = [e['kind'] for e in events]
    assert kinds[0] == 'run_started'
    assert kinds[-1] == 'run_finished'
    assert 'chapter_started' in kinds and 'chapter_finished' in kinds

    started_calls = [e['call'] for e in events if e['kind'] == 'call_started']
    finished_calls = {e['call']: e for e in events if e['kind'] == 'call_finished'}
    assert started_calls and set(started_calls) == set(finished_calls)
    for call in started_calls:
        assert finished_calls[call]['call'] == call

    operations = {e['operation'] for e in events if e['kind'] in ('call_started', 'call_finished')}
    assert {'naive', 'glossary', 'memory_extract', 'contextual'} <= operations

    out = tmp_path / job.id
    assert (out / 'results.json').is_file()
    found = wb.discover(tmp_path)
    run = next(r for r in found['runs'] if r['id'] == job.id)
    assert run['activity'] is True and run['replayable'] is True

    assert job.plan['api_key'] is None  # dropped from memory once the run ends
    _no_secret_on_disk(tmp_path, 'sk-test')


def test_jobboard_upload_run_orders_chapters_naturally_and_validates_inputs(tmp_path):
    board = wr.JobBoard(tmp_path)
    chapters = [dict(name='ch10.txt', text='林青躲开了箭。'), dict(name='ch2.txt', text='顾绫笑了。')]
    glossary = dict(name='terms.json', text=json.dumps([dict(source='林青', target='Lin Qing')]))
    job = board.start(dict(source='upload', provider='openai', model='test-model', api_key='sk-test',
                           compare=False, limit=0, chapters=chapters, glossary=glossary),
                      client=Client(), background=False)
    assert job.status == 'done'

    out = tmp_path / job.id
    uploads = tmp_path / wr.UPLOADS
    upload_dirs = [p for p in uploads.glob('*/chapters') if p.is_dir()]
    assert len(upload_dirs) == 1
    uploaded_names = sorted(p.name for p in upload_dirs[0].glob('*.txt'))
    assert uploaded_names == ['ch10.txt', 'ch2.txt']

    config = json.loads((out / 'config.json').read_text(encoding='utf-8'))
    ordered_names = [item['name'] for item in config['inputs']]
    assert ordered_names.index('ch2.txt') < ordered_names.index('ch10.txt')

    _no_secret_on_disk(tmp_path, 'sk-test')

    base = dict(source='upload', provider='openai', model='test-model', api_key='sk-test', compare=False)
    # save_uploads() is exercised directly for the per-chapter-file checks, each in its own
    # folder, since prepare() would otherwise reuse the same second-resolution stamp directory
    # across these quick, back-to-back calls and raise FileExistsError instead of ValueError.
    with pytest.raises(ValueError):
        wr.save_uploads(tmp_path / 'bad1', [dict(name='a.md', text='x')], None)
    with pytest.raises(ValueError):
        wr.save_uploads(tmp_path / 'bad2', [dict(name='a.txt', text='x'), dict(name='a.txt', text='y')], None)
    with pytest.raises(ValueError):
        wr.save_uploads(tmp_path / 'bad3', [], None)
    with pytest.raises(ValueError):
        wr.prepare(tmp_path, dict(base, model='', chapters=[dict(name='a.txt', text='x')]))
    with pytest.raises(ValueError):
        wr.prepare(tmp_path, dict(base, chapters=[]))

    monkeypatch_removed = os.environ.pop('OPENAI_API_KEY', None)
    try:
        with pytest.raises(ValueError):
            wr.prepare(tmp_path, dict(base, api_key=None, chapters=[dict(name='a.txt', text='x')]))
    finally:
        if monkeypatch_removed is not None:
            os.environ['OPENAI_API_KEY'] = monkeypatch_removed


def test_jobboard_replay_matches_original_with_zero_network(root):
    tmp_path, report = root
    board = wr.JobBoard(tmp_path)
    job = board.start(dict(source='replay', replay='run1'), background=False)
    assert job.status == 'done'

    events = wr.read_activity(tmp_path / job.id)
    run_started = next(e for e in events if e['kind'] == 'run_started')
    assert run_started['mode'] == 'cache replay'

    results = json.loads((tmp_path / job.id / 'results.json').read_text(encoding='utf-8'))
    original_texts = {c['id']: c['texts'] for c in report['chapters']}
    replayed_texts = {c['id']: c['texts'] for c in results['chapters']}
    assert replayed_texts == original_texts

    with pytest.raises(ValueError):
        board.start(dict(source='replay', replay='../x'), background=False)


class _StoppingClient(Client):
    """Stops the job after a fixed number of physical calls, via the board's active job."""

    def __init__(self, board, stop_after):
        super().__init__()
        self.board, self.stop_after, self.count = board, stop_after, 0

    def create(self, **kw):
        self.count += 1
        result = super().create(**kw)
        if self.count == self.stop_after:
            job = self.board.active()
            if job is not None:
                job.stop_requested = True
        return result


def test_jobboard_stop_ends_run_early_and_second_start_while_active_conflicts(tmp_path):
    unstopped_root = tmp_path / 'unstopped'
    unstopped_root.mkdir()
    unstopped_board = wr.JobBoard(unstopped_root)
    unstopped_job = unstopped_board.start(
        dict(source='sample', provider='openai', model='test-model', api_key='sk-test', compare=True, limit=1),
        client=Client(), background=False)
    assert unstopped_job.status == 'done'

    stopped_root = tmp_path / 'stopped'
    stopped_root.mkdir()
    stopped_board = wr.JobBoard(stopped_root)
    stopped_job = stopped_board.start(
        dict(source='sample', provider='openai', model='test-model', api_key='sk-test', compare=True, limit=2),
        client=_StoppingClient(stopped_board, stop_after=1), background=False)

    events = wr.read_activity(stopped_root / stopped_job.id)
    run_finished = next(e for e in events if e['kind'] == 'run_finished')
    assert run_finished['stopped'] is True
    assert stopped_job.status == 'stopped'
    assert stopped_job.calls < unstopped_job.calls

    conflict_root = tmp_path / 'conflict'
    conflict_root.mkdir()
    conflict_board = wr.JobBoard(conflict_root)
    ready, release = threading.Event(), threading.Event()

    class BlockingClient(Client):
        def create(self, **kw):
            ready.set()
            release.wait(timeout=5)
            return super().create(**kw)

    try:
        conflict_board.start(dict(source='sample', provider='openai', model='test-model', api_key='sk-test',
                                  compare=True, limit=1), client=BlockingClient(), background=True)
        assert ready.wait(timeout=5)
        with pytest.raises(RuntimeError):
            conflict_board.start(dict(source='sample', provider='openai', model='test-model', api_key='sk-test',
                                      compare=True, limit=1), client=Client(), background=True)
    finally:
        release.set()
        for _ in range(50):
            if not conflict_board.active():
                break
            time.sleep(0.05)


def test_start_stop_http_endpoints(tmp_path):
    board = wr.JobBoard(tmp_path)
    recorded = board.start(dict(source='sample', provider='openai', model='test-model', api_key='sk-test',
                                compare=True, limit=1), client=Client(), background=False)
    assert recorded.status == 'done'

    # serve() builds its own fresh JobBoard; the recorded run above is only tracked on disk
    # (via activity.jsonl), so /api/activity exercises the "recorded" (not in-memory) path.
    server = wb.serve(tmp_path, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    base = f'http://127.0.0.1:{port}'
    try:
        with urllib.request.urlopen(f'{base}/api/setup') as resp:
            raw = resp.read()
            setup = json.loads(raw)
            assert 'openai' in setup['providers']
            for info in setup['providers'].values():
                assert isinstance(info['key_set'], bool)
            assert b'sk-test' not in raw

        def post(path, *, body=None, origin=None, content_type='application/json'):
            data = json.dumps(body if body is not None else {}).encode()
            req = urllib.request.Request(f'{base}{path}', data=data, method='POST',
                                         headers={'Content-Type': content_type})
            if origin is not None:
                req.add_header('Origin', origin)
            return req

        try:
            urllib.request.urlopen(post('/api/start', body=dict(source='sample')))
            assert False, 'expected 403 with no Origin header'
        except urllib.error.HTTPError as err:
            assert err.code == 403

        try:
            urllib.request.urlopen(post('/api/start', body=dict(source='sample'), origin=f'http://127.0.0.1:{port + 1}'))
            assert False, 'expected 403 for wrong-port Origin'
        except urllib.error.HTTPError as err:
            assert err.code == 403

        try:
            urllib.request.urlopen(post('/api/start', body=dict(source='sample'), origin=base, content_type='text/plain'))
            assert False, 'expected 415 for non-JSON content type'
        except urllib.error.HTTPError as err:
            assert err.code == 415

        try:
            urllib.request.urlopen(post('/api/start', body=dict(source='sample'), origin=base))
            assert False, 'expected 400 for a missing model/key'
        except urllib.error.HTTPError as err:
            assert err.code == 400
            error = json.loads(err.read())
            assert isinstance(error.get('error'), str) and error['error']

        with urllib.request.urlopen(f'{base}/api/activity?id={recorded.id}') as resp:
            activity = json.loads(resp.read())
            assert activity['status'] == 'recorded'
            assert activity['events']

        with urllib.request.urlopen(post('/api/stop', origin=base)) as resp:
            stopped = json.loads(resp.read())
            assert stopped == {'id': None}
    finally:
        server.shutdown()
        server.server_close()


def test_endpoint_url_is_used_only_with_the_custom_provider(tmp_path):
    # A URL remembered from an earlier "custom" choice must not redirect a named provider's run.
    request = dict(source='sample', model='test-model', api_key='sk-test', base_url='https://leftover.example/v1')
    with pytest.raises(ValueError, match='custom provider'):
        wr.prepare(tmp_path, dict(request, provider='deepseek'))
    plan = wr.prepare(tmp_path, dict(request, provider='custom'))
    assert plan['config']['base_url'] == 'https://leftover.example/v1'


def test_activity_bookkeeping_errors_never_fail_a_call_or_the_run(tmp_path, monkeypatch):
    def unreadable(self, chapter_id):
        raise ValueError('result.json could not be read')
    monkeypatch.setattr(wr.Job, 'chapter_summary', unreadable)
    board = wr.JobBoard(tmp_path)
    job = board.start(dict(source='sample', provider='openai', model='test-model', api_key='sk-test',
                           compare=False, limit=2), client=Client(), background=False)
    assert job.status == 'done'
    events = job.activity.since(0)
    kinds = [e['kind'] for e in events]
    assert 'activity_error' in kinds and 'run_failed' not in kinds and kinds[-1] == 'run_finished'
    # The scripted client's term-alignment replies are invalid in every run; the injected
    # ValueError itself must never reach a model call.
    assert not any(e.get('error') == 'ValueError' for e in events if e['kind'] == 'call_finished')
    results = json.loads((tmp_path / job.id / 'results.json').read_text(encoding='utf-8'))
    assert len(results['chapters']) == 2 and not any(c['status'] == 'failed' for c in results['chapters'])
    finished = [e['chapter'] for e in events if e['kind'] == 'chapter_finished']
    assert finished == [c['id'] for c in results['chapters']]  # each chapter still closes in the feed



def test_failed_run_also_drops_the_pasted_key(tmp_path, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError('run_folder failed')
    monkeypatch.setattr(wr, 'run_folder', broken)
    job = wr.JobBoard(tmp_path).start(dict(source='sample', provider='openai', model='test-model', api_key='sk-test',
                                           limit=1), client=Client(), background=False)
    assert job.status == 'failed' and job.plan['api_key'] is None
    assert job.activity.since(0)[-1]['kind'] == 'run_failed'

def test_malformed_post_headers_are_answered_not_crashed(tmp_path):
    server = wb.serve(tmp_path, port=0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port = server.server_address[1]

    def post(headers):
        conn = http.client.HTTPConnection('127.0.0.1', port, timeout=5)
        try:
            conn.putrequest('POST', '/api/stop')
            for name, value in headers.items():
                conn.putheader(name, value)
            conn.endheaders()
            return conn.getresponse().status
        finally:
            conn.close()

    good = {'Origin': f'http://127.0.0.1:{port}', 'Content-Type': 'application/json'}
    try:
        assert post({**good, 'Content-Length': 'abc'}) == 400
        assert post({**good, 'Content-Length': '-1'}) == 400  # previously waited for the client to hang up
        assert post({**good, 'Origin': 'http://127.0.0.1:notaport', 'Content-Length': '0'}) == 403
        assert post({**good, 'Origin': 'http://[::1', 'Content-Length': '0'}) == 403
        assert post({**good, 'Content-Length': '0'}) == 200
    finally:
        server.shutdown()
        server.server_close()


def test_active_is_safe_while_another_request_adds_a_job(tmp_path):
    board = wr.JobBoard(tmp_path)
    board.jobs.update({f'old-{i}': NS(status='done') for i in range(10000)})
    errors, done = [], threading.Event()
    interval = sys.getswitchinterval()

    def add_jobs():
        for i in range(50):
            with board.lock:  # as start() does
                board.jobs[f'new-{i}'] = NS(status='done')
            time.sleep(0)
        done.set()

    sys.setswitchinterval(1e-5)  # switch threads often enough to land inside active()'s loop
    try:
        adder = threading.Thread(target=add_jobs)
        adder.start()
        while not done.is_set() and not errors:
            try:
                assert board.active() is None
            except RuntimeError as exc:  # "dictionary changed size during iteration"
                errors.append(exc)
        adder.join()
    finally:
        sys.setswitchinterval(interval)
    assert not errors


def test_legacy_report_template_escapes_chapter_and_model_text(tmp_path):
    from agentic_translation.report import render_report
    templates = tmp_path / 'templates'; templates.mkdir()
    (templates / 'report.html.j2').write_text('<pre>{{ source_text }}</pre>', encoding='utf-8')
    out = render_report(output_path=tmp_path / 'report.html', template_dir=templates,
                        context=dict(source_text='<script>steal()</script>'))
    assert out.read_text(encoding='utf-8') == '<pre>&lt;script&gt;steal()&lt;/script&gt;</pre>'


def test_run_files_symlinked_outside_the_runs_folder_are_not_read(root, tmp_path_factory):
    tmp_path, _ = root
    outside = tmp_path_factory.mktemp('outside') / 'secret.json'
    outside.write_text(json.dumps({'secret': 'outside'}), encoding='utf-8')
    (tmp_path / 'run1' / 'config.json').unlink()
    (tmp_path / 'run1' / 'config.json').symlink_to(outside)
    loaded = wb.load_run(tmp_path.resolve(), 'run1')
    assert loaded['config'] == {} and 'secret' not in json.dumps(loaded)


def test_one_malformed_results_file_does_not_hide_the_other_runs(root):
    tmp_path, _ = root
    (tmp_path / 'broken').mkdir()
    (tmp_path / 'broken' / 'results.json').write_text(json.dumps({'arms': [[]], 'chapters': []}), encoding='utf-8')
    ids = [run['id'] for run in wb.discover(tmp_path.resolve())['runs']]
    assert 'run1' in ids and 'broken' not in ids


def test_a_run_served_as_the_root_can_be_replayed_and_lists_new_runs(root):
    tmp_path, _ = root
    run = (tmp_path / 'run1').resolve()
    board = wr.JobBoard(run)
    assert [r['id'] for r in wb.setup(run, board)['replays']] == ['.']
    job = board.start(dict(source='replay', replay='.'), background=False)
    assert job.status == 'done' and job.id.startswith('workbench/')
    assert {r['id'] for r in wb.discover(run)['runs']} == {'.', job.id}


def test_upload_names_that_differ_only_in_case_are_refused_and_leave_nothing(tmp_path):
    request = dict(source='upload', provider='openai', model='test-model', api_key='sk-test',
                   chapters=[dict(name='Ch1.txt', text='第一章'), dict(name='ch1.txt', text='第二章')])
    with pytest.raises(ValueError, match='differ only in case'):
        wr.prepare(tmp_path, request)
    with pytest.raises(ValueError):
        wr.prepare(tmp_path, dict(request, chapters=[dict(name='1.txt', text='x' * 12001)]))
    assert not any((tmp_path / wr.UPLOADS).glob('*/*'))  # refused starts keep no chapter copies


def test_wrong_field_types_get_a_400_not_a_dropped_connection(tmp_path):
    server = wb.serve(tmp_path, port=0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f'http://127.0.0.1:{server.server_address[1]}'
    upload = dict(source='upload', model='test-model', api_key='sk-test', chapters=[dict(name='1.txt', text='x')])
    try:
        for body in (dict(upload, glossary='terms.txt'), dict(upload, provider=['openai']), dict(upload, base_url=5)):
            req = urllib.request.Request(f'{base}/api/start', data=json.dumps(body).encode(), method='POST',
                                         headers={'Content-Type': 'application/json', 'Origin': base})
            with pytest.raises(urllib.error.HTTPError) as err:
                urllib.request.urlopen(req, timeout=5)
            assert err.value.code == 400 and json.loads(err.value.read())['error']
    finally:
        server.shutdown()
        server.server_close()


class _CountingClient(Client):
    def __init__(self):
        super().__init__()
        self.count = 0

    def create(self, **kw):
        self.count += 1
        return super().create(**kw)


def test_stop_also_refuses_the_schema_correction_retry(tmp_path):
    board = wr.JobBoard(tmp_path)

    class StopThenInvalid(_CountingClient):
        def create(self, **kw):
            self.count += 1
            board.active().stop_requested = True  # Stop clicked while this request is in flight
            return NS(model='test-model', usage=NS(prompt_tokens=10, completion_tokens=5),
                      choices=[NS(finish_reason='stop', message=NS(content='not json'))])

    client = StopThenInvalid()
    job = board.start(dict(source='sample', provider='openai', model='test-model', api_key='sk-test',
                           compare=False, limit=1), client=client, background=False)
    assert client.count == 1  # no correction request after the stop
    assert job.status == 'stopped'


def test_a_stop_that_prevents_no_call_leaves_the_run_done(tmp_path):
    request = dict(source='sample', provider='openai', model='test-model', api_key='sk-test', compare=False, limit=1)
    counter = _CountingClient()
    (tmp_path / 'count').mkdir()
    wr.JobBoard(tmp_path / 'count').start(request, client=counter, background=False)
    (tmp_path / 'late').mkdir()
    board = wr.JobBoard(tmp_path / 'late')
    job = board.start(request, client=_StoppingClient(board, stop_after=counter.count), background=False)
    finished = job.activity.since(0)[-1]
    assert finished['kind'] == 'run_finished' and finished['stopped'] is False
    assert job.status == 'done'


def test_a_client_that_fails_to_close_still_frees_the_board_and_drops_the_key(tmp_path, monkeypatch):
    def broken_close(self):
        raise OSError('close failed')
    monkeypatch.setattr(wr.TrackedGenerator, 'close', broken_close)
    board = wr.JobBoard(tmp_path)
    with pytest.raises(OSError):
        board.start(dict(source='sample', provider='openai', model='test-model', api_key='sk-test',
                         compare=False, limit=1), client=Client(), background=False)
    job = next(iter(board.jobs.values()))
    assert job.status == 'done' and job.plan['api_key'] is None and board.active() is None
