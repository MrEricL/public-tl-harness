from pathlib import Path
from types import SimpleNamespace
import json
import pytest
from agentic_translation.autonomous_provider import BudgetedGenerator, GenerationOutputError, GenerationTransportError, ProviderAccessError, ledger_for


def config(tmp_path):
    return {"writer":{"model":"deepseek-flash"},"budgets":{"ledger":str(tmp_path/'budget.json'),"max_experiment_usd":10,"evaluation_reserve_usd":0}}


def response(content='{"ok":true}', finish='stop'):
    usage={"prompt_tokens":100,"completion_tokens":20,"prompt_cache_hit_tokens":50}
    return SimpleNamespace(model='deepseek-flash', usage=SimpleNamespace(model_dump=lambda:usage),
        choices=[SimpleNamespace(finish_reason=finish,message=SimpleNamespace(content=content))],
        model_dump=lambda:{"usage":usage,"content":content,"finish_reason":finish})


def install_client(monkeypatch, outputs):
    calls=[]
    class Fake:
        def __init__(self, **kwargs):
            assert kwargs['max_retries']==0
            self.chat=SimpleNamespace(completions=SimpleNamespace(create=self.create))
        def __enter__(self): return self
        def __exit__(self,*args): pass
        def create(self, **kwargs):
            calls.append(kwargs)
            item=outputs.pop(0)
            if isinstance(item,Exception): raise item
            return item
    monkeypatch.setattr('openai.OpenAI',Fake)
    monkeypatch.setenv('DEEPSEEK_API_KEY','test')
    monkeypatch.setattr('agentic_translation.autonomous_provider.time.sleep',lambda _:None)
    return calls

SCHEMA={"type":"object","properties":{"ok":{"type":"boolean"}},"required":["ok"],"additionalProperties":False}


def test_live_cache_preserves_logical_usage_without_rebilling(tmp_path,monkeypatch):
    calls=install_client(monkeypatch,[response()])
    cfg=config(tmp_path)
    provider=BudgetedGenerator(tmp_path/'run',cfg)
    assert provider.call('naive','test',SCHEMA)=={'ok':True}
    snapshot=ledger_for(cfg).snapshot()
    assert snapshot['unknown_usage_calls']==0
    provider2=BudgetedGenerator(tmp_path/'run',cfg,replay=True)
    assert provider2.call('naive','test',SCHEMA)=={'ok':True}
    assert len(calls)==1
    assert provider2.receipts[0]['estimated_charge']==provider.receipts[0]['estimated_charge']
    assert provider2.receipts[0]['physical_charge_usd']==0
    assert ledger_for(cfg).snapshot()['budget_committed_usd']==snapshot['budget_committed_usd']


def test_truncation_attempt_is_billed_before_uniform_recovery(tmp_path,monkeypatch):
    calls=install_client(monkeypatch,[response(finish='length'),response()])
    cfg=config(tmp_path)
    provider=BudgetedGenerator(tmp_path/'run',cfg)
    assert provider.call('contextual','test',SCHEMA)=={'ok':True}
    assert len(calls)==2 and len(provider.receipts)==2
    assert provider.receipts[0]['status']=='failed'
    assert provider.receipts[0]['estimated_charge'] is not None


def test_exhausted_critique_output_is_nonfatal_and_billed(tmp_path, monkeypatch):
    calls = install_client(monkeypatch, [response(finish='length'), response(finish='length')])
    cfg = config(tmp_path)
    provider = BudgetedGenerator(tmp_path/'run', cfg)
    with pytest.raises(GenerationOutputError, match='output exhausted'):
        provider.call('simple_critique', 'Find errors only', SCHEMA, max_output_tokens=4096)
    assert len(calls) == 2
    assert all(receipt['estimated_charge'] is not None for receipt in provider.receipts)
    assert provider.receipts[-1]['failure_kind'] == 'output_exhausted'
    assert 'at most TWELVE actual material errors' in calls[1]['messages'][-1]['content']


def test_exhausted_transport_is_fatal_even_after_retry(tmp_path, monkeypatch):
    class APIConnectionError(Exception):
        pass

    calls = install_client(monkeypatch, [APIConnectionError(), APIConnectionError(), APIConnectionError()])
    provider = BudgetedGenerator(tmp_path/'run', config(tmp_path))
    with pytest.raises(GenerationTransportError, match='APIConnectionError'):
        provider.call('simple_critique', 'Find errors only', SCHEMA)
    assert len(calls) == 3
    assert all(receipt['failure_kind'] == 'provider_or_transport' for receipt in provider.receipts)
    assert all(receipt['estimated_charge'] is None for receipt in provider.receipts)


def test_balance_failure_never_retries_or_marks_request_free(tmp_path,monkeypatch):
    class BalanceError(Exception): status_code=402
    calls=install_client(monkeypatch,[BalanceError()])
    cfg=config(tmp_path)
    provider=BudgetedGenerator(tmp_path/'run',cfg)
    with pytest.raises(ProviderAccessError): provider.call('naive','test',SCHEMA)
    assert len(calls)==1
    assert ledger_for(cfg).snapshot()['unknown_usage_calls']==1


def test_segment_text_envelope_preserves_dialogue_and_detects_missing_ids(tmp_path,monkeypatch):
    from agentic_translation.autonomous_provider import parse_segment_text
    raw='<<<SEGMENT:s0001>>>\n"Enough," she said. \\n is literal.\n\n<<<SEGMENT:s0002>>>\nHe answered.\n'
    payload=parse_segment_text(raw,['s0001','s0002'])
    assert payload['segments'][0]['translated_text']=='"Enough," she said. \\n is literal.'
    with pytest.raises(ValueError,match='exactly once'): parse_segment_text(raw,['s0001','s0002','s0003'])
    calls=install_client(monkeypatch,[response(raw)])
    schema={'type':'object','properties':{'segments':{'type':'array'}},'required':['segments'],'x-segment-text-envelope':['s0001','s0002']}
    p=BudgetedGenerator(tmp_path/'run',config(tmp_path))
    assert p.call('naive','Translate',schema)==payload
    assert 'response_format' not in calls[0]


def test_memory_scope_alias_is_canonicalized_without_rewriting_fact(tmp_path,monkeypatch):
    payload={'entries':[{'kind':'dialogue_claim','scope':'dialogue_claim','value':'The speaker claims to be the heir.'}]}
    install_client(monkeypatch,[response(json.dumps(payload))])
    schema={'type':'object','properties':{'entries':{'type':'array','items':{'type':'object','properties':{'kind':{'enum':['state']}}}}}}
    p=BudgetedGenerator(tmp_path/'run',config(tmp_path))
    result=p.call('memory_extract','Extract source facts',schema)
    assert result['entries'][0]=={'kind':'state','scope':'dialogue_claim','value':'The speaker claims to be the heir.'}
    assert p.receipts[0]['response_normalizations'][0]['from']=='dialogue_claim'
    raw=json.loads(next((tmp_path/'run'/'calls').glob('*.json')).read_text())
    assert 'dialogue_claim' in raw['raw_response']['content']


def test_judge_factory_defaults_to_claude_and_luna_with_no_adjudicator(tmp_path):
    from agentic_translation.autonomous_provider import make_judges
    cfg = config(tmp_path)
    primaries, adjudicator = make_judges(tmp_path, cfg)
    assert [(model, family) for model, family, _call in primaries] == [
        ("claude-sonnet-5", "anthropic"), ("gpt-6-luna", "openai")]
    assert adjudicator is None


def test_judge_factory_new_judges_list_shape_honors_per_judge_settings(tmp_path, monkeypatch):
    from agentic_translation.autonomous_provider import make_judges
    calls = []

    def fake_invoke(spec, prompt, schema, folder, *, provider_mode=None):
        calls.append((spec, provider_mode))
        return {"status": "completed", "output": {"preference": "tie"}, "usage": {"output_tokens": 5}}

    monkeypatch.setattr("agentic_translation.structured_transport.invoke_structured", fake_invoke)
    cfg = config(tmp_path)
    cfg["judging"] = {"judges": [
        {"model": "claude-sonnet-5", "backend": "claude", "family": "anthropic",
         "effort": "medium", "max_output_tokens": 16384, "timeout_seconds": 600},
        {"model": "gpt-6-luna", "backend": "codex", "family": "openai",
         "effort": "high", "max_output_tokens": 16384, "timeout_seconds": 600},
    ]}
    primaries, adjudicator = make_judges(tmp_path, cfg)
    assert adjudicator is None
    request = {"prompt": "judge", "schema": {"type": "object", "properties": {"preference": {"type": "string"}}}}
    assert primaries[0][2](request)["preference"] == "tie"
    assert primaries[1][2](request)["preference"] == "tie"
    assert [(spec.backend, spec.model, spec.effort, spec.max_output_tokens, spec.timeout_seconds) for spec, _ in calls] == [
        ("claude", "claude-sonnet-5", "medium", 16384, 600),
        ("codex", "gpt-6-luna", "high", 16384, 600),
    ]
    assert all(mode == "replay" for _, mode in calls)


def test_judge_factory_honors_old_keys_and_rejects_false_family_labels(tmp_path):
    from agentic_translation.autonomous_provider import make_judges
    cfg=config(tmp_path)
    cfg['judging']={'primary_models':['deepseek-flash','gpt-5.6-luna'], 'primary_families':['deepseek','openai'], 'adjudicator_model':'gpt-5.6-terra'}
    primaries, adjudicator=make_judges(tmp_path,cfg)
    assert [x[0] for x in primaries]==cfg['judging']['primary_models']
    assert adjudicator[0]=='gpt-5.6-terra'
    cfg['judging']['primary_families']=['openai','openai']
    with pytest.raises(ValueError,match='families'): make_judges(tmp_path,cfg)


def test_judge_cli_roles_use_configured_effort_token_and_time_limits(tmp_path, monkeypatch):
    from agentic_translation.autonomous_provider import make_judges
    calls = []

    def fake_invoke(spec, prompt, schema, folder, *, provider_mode=None):
        calls.append((spec, provider_mode))
        return {"status": "completed", "output": {"preference": "tie"}}

    monkeypatch.setattr("agentic_translation.structured_transport.invoke_structured", fake_invoke)
    cfg = config(tmp_path)
    cfg["judging"] = {
        "primary_models": ["deepseek-v4-pro", "gpt-5.6-luna"],
        "primary_families": ["deepseek", "openai"],
        "adjudicator_model": "gpt-5.6-terra",
        "cli_primary_effort": "medium", "cli_primary_max_output_tokens": 16384,
        "cli_primary_timeout_seconds": 600,
        "cli_adjudicator_effort": "high", "cli_adjudicator_max_output_tokens": 16384,
        "cli_adjudicator_timeout_seconds": 600,
    }
    primaries, adjudicator = make_judges(tmp_path, cfg)
    request = {"prompt": "judge", "schema": {"type": "object", "properties": {"preference": {"type": "string"}}}}
    assert primaries[1][2](request)["preference"] == "tie"
    assert adjudicator[2](request)["preference"] == "tie"
    assert [(spec.model, spec.effort, spec.max_output_tokens, spec.timeout_seconds) for spec, _ in calls] == [
        ("gpt-5.6-luna", "medium", 16384, 600),
        ("gpt-5.6-terra", "high", 16384, 600),
    ]
    assert all(mode == "replay" for _, mode in calls)


def test_judge_cli_defaults_are_usable_for_existing_resolved_config(tmp_path):
    from agentic_translation.autonomous_provider import make_judges
    cfg = config(tmp_path)
    cfg["judging"] = {"primary_models": ["deepseek-v4-pro", "gpt-5.6-luna"],
                      "adjudicator_model": "gpt-5.6-terra"}
    primaries, adjudicator = make_judges(tmp_path, cfg)
    assert primaries[1][:2] == ("gpt-5.6-luna", "openai")
    assert adjudicator[:2] == ("gpt-5.6-terra", "openai")


def test_claude_judge_records_subscription_equivalent_cost_not_charged(tmp_path, monkeypatch):
    from agentic_translation.autonomous_provider import make_judges

    def fake_invoke(spec, prompt, schema, folder, *, provider_mode=None):
        assert spec.backend == "claude"
        return {"status": "completed", "output": {"preference": "A"},
                "usage": {"input_tokens": 10, "output_tokens": 2},
                "cli_reported_cost_usd": 0.0456}

    monkeypatch.setattr("agentic_translation.structured_transport.invoke_structured", fake_invoke)
    cfg = config(tmp_path)
    cfg["judging"] = {"judges": [{"model": "claude-sonnet-5", "backend": "claude",
                                  "family": "anthropic", "effort": "medium"}]}
    primaries, _adjudicator = make_judges(tmp_path, cfg)
    result = primaries[0][2]({"prompt": "judge", "schema": {"type": "object", "properties": {}}})
    assert result["receipt"]["estimated_charge"] is None
    assert result["receipt"]["subscription_equivalent_usd"] == pytest.approx(0.0456)
    assert result["receipt"]["cost_basis"] == "subscription_equivalent_usd_not_charged"


def test_codex_judge_records_token_usage_from_receipt(tmp_path, monkeypatch):
    from agentic_translation.autonomous_provider import make_judges

    def fake_invoke(spec, prompt, schema, folder, *, provider_mode=None):
        assert spec.backend == "codex"
        return {"status": "completed", "output": {"preference": "B"},
                "usage": {"input_tokens": 100, "output_tokens": 50,
                          "cached_input_tokens": 0, "total_tokens": 150}}

    monkeypatch.setattr("agentic_translation.structured_transport.invoke_structured", fake_invoke)
    cfg = config(tmp_path)
    cfg["judging"] = {"judges": [{"model": "gpt-6-luna", "backend": "codex",
                                  "family": "openai", "effort": "high"}]}
    primaries, _adjudicator = make_judges(tmp_path, cfg)
    result = primaries[0][2]({"prompt": "judge", "schema": {"type": "object", "properties": {}}})
    assert result["receipt"]["estimated_charge"] is None
    assert result["receipt"]["usage"]["output_tokens"] == 50
    assert result["receipt"]["cost_basis"] == "subscription_cli_marginal_cost_unavailable"
