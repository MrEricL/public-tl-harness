from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from agentic_translation.adaptive import _segment_deterministic_findings
from agentic_translation.demo import demo_policy, discover_sources, load_contract, main, run_folder
from agentic_translation.demo_metrics import resources, summarize, term_checks
from agentic_translation.demo_provider import DemoGenerator, digest, provider_config
from agentic_translation.models import GlossaryEntry, GlossaryParseResult
from agentic_translation.segmentation import parse_draft_envelope, segment_source
from agentic_translation.autonomous_provider import GenerationOutputError, GenerationTransportError
from experiments.portfolio_demo.repair_bench import CASES, checks, run_benchmark


class Client:
    """Scripted transport fixture, never a claim about a model's intelligence."""
    def __init__(self, responder=None):
        self.chat = NS(completions=self)
        self.calls = []
        self.responder = responder

    def create(self, **kw):
        self.calls.append(kw)
        if self.responder:
            return self.responder(kw)
        user = kw['messages'][1]['content']
        payload = json.JSONDecoder().raw_decode(user[user.index('\n{')+1:])[0]
        if user.startswith('Extract'):
            raw = json.dumps(dict(entries=[], terms=[]))
        elif user.startswith('Translate'):
            raw = '\n\n'.join(f'<<<SEGMENT:{row["segment_id"]}>>>\nLin Qing used Lunar Shadow Footwork to dodge the arrow.'
                               for row in payload['source_segments'])
        elif user.startswith('Each item'):
            raw = json.dumps({'patches': [dict(segment_id=row['segment_id'], expected_segment_sha256=row['expected_segment_sha256'],
                                                edits=[dict(old_text='Lunar Shadow Footwork', new_text='Moon-Shadow Step')],
                                                issue_ids=['term_alignment'], rationale='Apply the explicit glossary')
                                         for row in payload['segments']]})
        else:
            raw = json.dumps(dict(summary='No material issue in this fixture', findings=[], patches=[]))
        return answer(raw)


def answer(raw, reason='stop', usage=True):
    return NS(model='test-model', usage=NS(prompt_tokens=40, completion_tokens=30) if usage else None,
              choices=[NS(finish_reason=reason, message=NS(content=raw))])


@pytest.fixture
def inputs(tmp_path):
    source = tmp_path/'source'; source.mkdir()
    (source/'chapter2.txt').write_text('林青使用月影步躲开了箭。', encoding='utf-8')
    (source/'chapter10.txt').write_text('林青再次使用月影步躲开箭。', encoding='utf-8')
    glossary = tmp_path/'glossary.json'
    glossary.write_text(json.dumps([dict(source='月影步', target='Moon-Shadow Step', category='techniques',
                                         blocked_variants=['Lunar Shadow Footwork'])]), encoding='utf-8')
    return source, glossary


def test_folder_end_to_end_and_cache_replay(tmp_path, inputs, monkeypatch):
    source, glossary = inputs
    out = tmp_path/'live'; config = provider_config('openai', 'test-model')
    client = Client()
    gen = DemoGenerator(out, config, client=client)
    gen.evidence_mode = 'scripted integration fixture'
    report = run_folder(source, out, config, glossary_path=glossary, compare=True, generator=gen)
    assert report['status'] == 'delivered'
    assert report['arms'] == ['naive', 'glossary', 'contextual', 'harness']
    assert report['summary']['groups']['all']['harness']['passed'] == 2
    assert report['summary']['groups']['all']['contextual']['passed'] == 0
    assert report['summary']['gains'][-1]['regressed'] == 0
    assert report['resources']['estimated_usd'] is None
    assert (out/'delivery/book.epub').exists()
    assert [r['name'] for r in report['chapters']] == ['chapter2.txt', 'chapter10.txt']
    assert json.loads((out/'config.json').read_text())['profile'] == asdict(demo_policy())
    # Real transport must not be constructed during replay; no API key available.
    monkeypatch.setattr(DemoGenerator, '_transport', lambda self: pytest.fail('network during replay'))
    replay = tmp_path/'replayed'
    assert main(['--replay', str(out), '--out', str(replay)]) == 0
    repeated = json.loads((replay/'results.json').read_text())
    assert repeated['mode'] == 'cache replay'
    assert repeated['resources']['physical_calls'] == 0
    assert repeated['chapters'][0]['texts'] == report['chapters'][0]['texts']
    assert (replay/'delivery/book.txt').read_bytes() == (out/'delivery/book.txt').read_bytes()
    assert not (replay/'calls').exists()


# A fake key canary, split so release scanning does not mistake it for a credential.
CANARY_KEY = 'sk-' + 'this-must-not-be-logged'


def test_failures_remain_in_assigned_denominator(tmp_path, inputs):
    source, glossary = inputs; out=tmp_path/'failed'
    def fail(kw):
        raise RuntimeError(CANARY_KEY)
    gen=DemoGenerator(out, provider_config('openai','test-model'), client=Client(fail))
    gen.evidence_mode='scripted failure fixture'
    report=run_folder(source,out,gen.config,glossary_path=glossary,compare=True,generator=gen)
    assert report['status']=='failed_partial'
    assert len(report['chapters'])==2
    assert report['summary']['groups']['all']['harness']==dict(passed=0,total=2,rate=0)
    assert report['resources']['physical_calls']==1
    assert not (out/'delivery').exists()
    assert CANARY_KEY not in ''.join(p.read_text() for p in out.rglob('*.json'))


@pytest.mark.parametrize('url', ['http://example.com/v1','https://x?key=abc','https://key:secret@x/v1','https://x#token',''])
def test_unsafe_provider_urls_rejected(url):
    with pytest.raises(ValueError): provider_config('custom','x',base_url=url)


@pytest.mark.parametrize('arg,value', [('max_calls',0),('max_calls',True),('max_output_tokens',0),
                                     ('temperature',float('nan')),('temperature',3),('input_price',-1),
                                     ('max_usd',float('inf')),('call_timeout_seconds',0),
                                     ('call_timeout_seconds',float('nan')),('call_timeout_seconds',True)])
def test_invalid_config(arg,value):
    with pytest.raises(ValueError): provider_config('openai','x',**{arg:value})


def test_custom_endpoint_does_not_receive_another_providers_key(monkeypatch,tmp_path):
    monkeypatch.setenv('OPENAI_API_KEY','secret')
    monkeypatch.delenv('TL_API_KEY',raising=False)
    cfg=provider_config('openai','x',base_url='https://example.com/v1')
    with pytest.raises(ValueError, match='TL_API_KEY'): DemoGenerator(tmp_path,cfg)


def test_unknown_cost_and_physical_call_ceiling(tmp_path):
    cfg=provider_config('openai','x',max_calls=1)
    client=Client(lambda kw: answer('{"ok":true}', usage=False))
    gen=DemoGenerator(tmp_path,cfg,client=client)
    schema={'type':'object','properties':{'ok':{'type':'boolean'}},'required':['ok']}
    assert gen.call('one','prompt',schema)=={'ok':True}
    assert gen.receipts[0]['estimated_usd'] is None
    # A cache hit is free of physical calls, not assigned zero historical usage.
    gen.call('one','prompt',schema)
    assert len(client.calls)==1 and gen.receipts[-1]['physical_calls']==0
    with pytest.raises(GenerationTransportError): gen.call('two','prompt',schema)
    assert len(client.calls)==1


def test_estimated_budget_preflight(tmp_path):
    cfg=provider_config('openai','x',input_price=10,output_price=10,max_usd=0)
    client=Client()
    gen=DemoGenerator(tmp_path,cfg,client=client)
    with pytest.raises(GenerationTransportError,match='ceiling'): gen.call('x','x',{})
    assert not client.calls


@pytest.mark.parametrize('raw,reason', [('not-json','stop'),('{"ok":true}','length'),('{"ok":true}','content_filter')])
def test_malformed_and_truncated_outputs_are_never_success(tmp_path,raw,reason):
    client=Client(lambda kw: answer(raw,reason))
    gen=DemoGenerator(tmp_path,provider_config('openai','x'),client=client)
    with pytest.raises(GenerationOutputError): gen.call('x','x',{'type':'object'})
    assert gen.receipts[0]['status']=='failed'
    assert len(list((tmp_path/'calls').glob('*.json')))==1
    assert not (tmp_path/'cache').exists()


@pytest.mark.parametrize('provider,token_field', [('openai','max_completion_tokens'),('anthropic','max_tokens'),('deepseek','max_tokens'),('openrouter','max_tokens')])
def test_provider_request_shape(provider,token_field,tmp_path):
    client=Client(lambda kw: answer('{}'))
    gen=DemoGenerator(tmp_path,provider_config(provider,'test-model'),client=client)
    gen.call('x','x',{'type':'object'})
    assert token_field in client.calls[0] and 'temperature' not in client.calls[0]
    assert 'response_format' not in client.calls[0]


def test_overlap_and_case_contracts(tmp_path):
    glossary=GlossaryParseResult(entries=[GlossaryEntry(source='灵石',target='Spirit Stone'),
                                          GlossaryEntry(source='上品灵石',target='High-Grade Stone')])
    sources=segment_source('p','1','上品灵石。')
    drafts=parse_draft_envelope([dict(segment_id=sources[0].segment_id,translated_text='High-Grade Stone.')],sources)
    rows=term_checks(sources,drafts,glossary,{})
    assert len(rows)==1 and rows[0]['passed']
    assert not _segment_deterministic_findings('上品灵石。','High-Grade Stone.',glossary)
    e=GlossaryParseResult(entries=[GlossaryEntry(source='伊文',target='Evan')])
    assert _segment_deterministic_findings('伊文。','Evander.',e)


def test_empty_glossary_does_not_fabricate_100_percent():
    chapters=[dict(id='1',arms={a:dict(checks=[]) for a in ['contextual','harness']})]
    assert summarize(chapters,['contextual','harness'],metric='terms')['groups']=={}


def test_categorized_negative_effects_are_visible():
    def arm(p):return dict(checks=[dict(category='people',passed=p),dict(category='places',passed=not p)])
    r=summarize([dict(id='1',arms={'before':arm(True),'after':arm(False)})],['before','after'],metric='terms')
    assert next(g for g in r['gains'] if g['category']=='people')['percentage_points']==-100
    assert next(g for g in r['gains'] if g['category']=='people')['regressed']==1
    assert next(g for g in r['gains'] if g['category']=='places')['fixed']==1


@pytest.mark.parametrize('content', ['[{}]','[{"source":"x","target":"y"},{"source":"x","target":"z"}]',
                                    '[{"source":"x","target":""}]','[{"source":"x","target":"y","typo":true}]'])
def test_bad_glossary(content,tmp_path):
    p=tmp_path/'bad.json';p.write_text(content)
    with pytest.raises(ValueError):load_contract(p)


def test_input_validation_before_spending_and_no_overwrite(tmp_path,inputs):
    source,glossary=inputs
    with pytest.raises(ValueError,match='separate'):discover_sources(source,source/'out',0,12000)
    with pytest.raises(ValueError,match='exceeds'):discover_sources(source,tmp_path/'out',0,1)
    rows,excluded=discover_sources(source,tmp_path/'out',1,12000)
    assert len(rows)==1 and excluded==['chapter10.txt']
    out=tmp_path/'out';out.mkdir();(out/'keep.txt').write_text('safe')
    with pytest.raises(ValueError,match='empty'):run_folder(source,out,provider_config('openai','x'),glossary_path=glossary)
    assert (out/'keep.txt').read_text()=='safe'


def test_repair_fixture_rubric_is_self_consistent():
    cases=json.loads(CASES.read_text())
    assert len(cases)==12
    for case in cases:
        assert all(r['passed'] for r in checks(case,case['reference'])),case['id']
        assert all(r['passed'] for r in checks(case,case['draft'])) == bool(case.get('clean_control')),case['id']
    assert sum(c.get('clean_control',False) for c in cases)==2


def test_report_escapes_untrusted_content(tmp_path,inputs):
    source,glossary=inputs
    evil=source/'<script>alert(1)</script>.txt'
    # The literal filename cannot contain a slash; source text is sufficient.
    (source/'chapter2.txt').write_text('林青使用月影步躲开了箭。 <script>alert(1)</script>')
    out=tmp_path/'report';g=DemoGenerator(out,provider_config('openai','x'),client=Client())
    run_folder(source,out,g.config,glossary_path=glossary,generator=g)
    text=(out/'report.html').read_text()
    assert '<script>' not in text and '&lt;script&gt;' in text


def test_repair_benchmark_uses_same_real_executor_without_gold_in_requests(tmp_path):
    cases=json.loads(CASES.read_text())
    def responder(kw):
        prompt=kw['messages'][1]['content']
        assert '"checks"' not in prompt and '"reference"' not in prompt
        raw=json.JSONDecoder().raw_decode(prompt[prompt.index('\n{')+1:])[0]
        if prompt.startswith('Each item'):
            return answer(json.dumps({'patches':[]}))  # exercise reviewer path for terminology too
        case=next(c for c in cases if c['source']==raw['source_text'])
        old=raw['draft_text']
        if old==case['reference']:
            return answer(json.dumps(dict(summary='Good fixture',findings=[],patches=[])))
        import hashlib
        # Emulate the bounded edit contract, not an inadmissible whole-chapter rewrite.
        desired = case['reference']
        start = 0
        while start < min(len(old), len(desired)) and old[start] == desired[start]:
            start += 1
        end = 0
        while end < min(len(old)-start, len(desired)-start) and old[-end-1] == desired[-end-1]:
            end += 1
        if start == len(old)-end:  # insertion: include a small existing anchor
            start = max(0, start-4)
        old_span, new_span = old[start:len(old)-end or None], desired[start:len(desired)-end or None]
        patch=dict(segment_id=raw['segment_id'],expected_segment_sha256=hashlib.sha256(old.encode()).hexdigest(),
                   edits=[dict(old_text=old_span,new_text=new_span)],issue_ids=['material'],rationale='Correct the source fact')
        return answer(json.dumps(dict(summary='Material correction',
                                      findings=[dict(issue_id='material',severity='material',message='Changed source fact',blocking=True)],patches=[patch])))
    client=Client(responder);out=tmp_path/'bench'
    gen=DemoGenerator(out,provider_config('openai','test-model'),client=client)
    gen.evidence_mode='scripted integration fixture'
    report=run_benchmark(out,gen.config,generator=gen)
    assert report['profile']==asdict(demo_policy())
    assert report['summary']['groups']['all']['supplied_draft']['passed']==2
    assert report['summary']['groups']['all']['harness']['passed']==12
    assert report['summary']['clean_controls']==dict(unchanged=2,total=2)
    assert report['mode']=='scripted integration fixture'


def test_real_openai_sdk_against_local_http_endpoint(tmp_path):
    from http.server import BaseHTTPRequestHandler, HTTPServer
    import threading
    captured=[]
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            captured.append((self.path,json.loads(self.rfile.read(int(self.headers['Content-Length'])))))
            raw=json.dumps(dict(id='test',object='chat.completion',created=0,model='test-model',
                                 choices=[dict(index=0,finish_reason='stop',message=dict(role='assistant',content='{"ok":true}'))],
                                 usage=dict(prompt_tokens=5,completion_tokens=3,total_tokens=8))).encode()
            self.send_response(200);self.send_header('Content-Type','application/json');self.end_headers();self.wfile.write(raw)
        def log_message(self,*args): pass
    server=HTTPServer(('127.0.0.1',0),Handler)
    t=threading.Thread(target=server.serve_forever,daemon=True);t.start()
    try:
        cfg=provider_config('custom','test-model',base_url=f'http://127.0.0.1:{server.server_port}/v1',input_price=1,output_price=2)
        gen=DemoGenerator(tmp_path,cfg,api_key='not-a-real-key')
        assert gen.call('protocol','test',{'type':'object'})=={'ok':True}
        assert captured[0][0]=='/v1/chat/completions'
        assert gen.receipts[0]['estimated_usd']==pytest.approx(11/1e6)
        assert 'not-a-real-key' not in ''.join(p.read_text() for p in tmp_path.rglob('*.json'))
        gen.close()
    finally:
        server.shutdown();server.server_close();t.join(timeout=3)


def test_packaging_failure_keeps_translation_and_report(monkeypatch, tmp_path, inputs):
    import agentic_translation.demo as demo
    source, glossary = inputs
    out = tmp_path/'bad-epub'
    gen = DemoGenerator(out, provider_config('openai', 'test-model'), client=Client())
    gen.evidence_mode = 'scripted packaging failure fixture'
    def fail(**kwargs):
        raise OSError('export failure')
    monkeypatch.setattr(demo, 'build_epub_collection', fail)
    report = run_folder(source, out, gen.config, glossary_path=glossary, generator=gen)
    assert report['status'] == 'failed_packaging'
    assert report['delivery'] is False
    assert report['packaging_error_type'] == 'OSError'
    assert (out/'delivery/book.txt').read_text()
    assert (out/'report.html').exists()
    assert report['summary']['groups']['all']['harness']['passed'] == 2


def test_report_headline_and_clean_control_accounting(tmp_path, inputs):
    from agentic_translation.demo_metrics import write_report
    report = dict(mode='authored test', description='test', status='completed', delivery=False,
                  arms=['before','harness'], chapters=[],
                  summary=dict(metric='test checks', note='narrow metric', gains=[],
                               groups={'all': {'before': {'passed':2,'total':4,'rate':.5},
                                               'harness': {'passed':3,'total':4,'rate':.75}}},
                               clean_controls={'unchanged':1,'total':2}))
    write_report(tmp_path,report)
    text = (tmp_path/'report.html').read_text()
    assert '+25.0 percentage points (+50.0% relative)' in text
    assert 'Clean controls left unchanged: 1/2' in text


def test_queued_call_fails_at_the_wall_clock_deadline(tmp_path):
    """A provider that holds a request open (e.g. queue keep-alives) cannot hang the demo."""
    import threading, time
    release = threading.Event()
    def stall(kw):
        release.wait(5)
        return answer('{}')
    config = provider_config('openai', 'test-model', call_timeout_seconds=0.2)
    gen = DemoGenerator(tmp_path/'slow', config, client=Client(stall))
    started = time.perf_counter()
    try:
        with pytest.raises(GenerationTransportError, match='0.2-second deadline'):
            gen.call('probe', 'Probe', {'type': 'object'})
    finally:
        release.set()
    assert time.perf_counter() - started < 2
    assert gen.receipts[-1]['status'] == 'failed'
    assert gen.receipts[-1]['error_type'] == 'TimeoutError'


def test_failed_run_reports_no_score_instead_of_a_delta(tmp_path, inputs):
    source, glossary = inputs; out = tmp_path/'failed-headline'
    def fail(kw):
        raise RuntimeError('provider unavailable')
    gen = DemoGenerator(out, provider_config('openai', 'test-model'), client=Client(fail))
    gen.evidence_mode = 'scripted failure fixture'
    report = run_folder(source, out, gen.config, glossary_path=glossary, compare=True, generator=gen)
    assert report['status'] == 'failed_partial'
    text = (out/'report.md').read_text()
    assert 'No score: 2 of 2 item(s) lack output' in text
    assert 'percentage points' not in text.split('## Incremental effects')[0]
