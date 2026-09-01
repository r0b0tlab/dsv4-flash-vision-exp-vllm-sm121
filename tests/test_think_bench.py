import importlib.util, pathlib
spec = importlib.util.spec_from_file_location("tb", pathlib.Path(__file__).parents[1] / "scripts" / "think_bench.py")
tb = importlib.util.module_from_spec(spec); spec.loader.exec_module(tb)

def test_summarize_aggregate_and_streams():
    res = [{"dt": 10.0, "completion_tokens": 300, "has_reasoning": True, "finish": "stop"},
           {"dt": 20.0, "completion_tokens": 200, "has_reasoning": False, "finish": "length"}]
    s = tb.summarize(res, wall=20.0)
    assert s["aggregate_out_tok_s"] == 25.0
    assert s["per_stream_tok_s"] == {"min": 10.0, "med": 30.0, "max": 30.0}
    assert s["mean_completion_tokens"] == 250.0
    assert s["reasoning_present_frac"] == 0.5
    assert s["finish_reasons"] == ["length", "stop"]

def test_prompt_bank_is_16_unique():
    assert len(tb.PROMPTS) == 16 and len(set(tb.PROMPTS)) == 16
