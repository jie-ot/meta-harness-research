"""No-network acceptance of data, traces, cache, phases, recovery and prediction."""
from __future__ import annotations

import contextlib
import io
import json
import os
import random
import shutil
import sys
import tempfile
import time
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

from pilot.state import ROOT, atomic_json, digest, jsonl_write, read_json, read_jsonl

sys.path.insert(0, str(ROOT / "engine"))
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
from text_classification.data.api import load_pilot_split
from text_classification.data.loaders import wrap_lawbench_rows
from text_classification.inner_loop import evaluate_memory
from text_classification.llm import LLM, ProviderLLM
from text_classification.memory_system import MemorySystem
from text_classification.pilot_observation import capture_calls, record_call, set_event_callback


@contextlib.contextmanager
def scratch():
    parent = ROOT / ".verification_tmp"
    parent.mkdir(exist_ok=True)
    directory = parent / uuid.uuid4().hex
    directory.mkdir()
    try:
        yield directory
    finally:
        resolved = directory.resolve()
        if not resolved.is_relative_to(parent.resolve()) or resolved == parent.resolve():
            raise ValueError("Unsafe test cleanup path")
        shutil.rmtree(resolved)


def fake_response(prompt, system_prompt, kwargs):
    return {"content": prompt, "input_tokens": 2, "output_tokens": 1, "cost": .001,
            "reported_usd": .001, "estimated_usd": None, "cost_source": "synthetic_test_only",
            "raw_response": {"synthetic_test_only": True, "content": prompt}}


class CountingMemory(MemorySystem):
    def __init__(self, llm):
        super().__init__(llm)
        self.learned = 200
    def predict(self, input):
        return self.call_llm(input), {"preserved": "metadata"}
    def learn_from_batch(self, batch_results):
        self.learned += len(batch_results)
    def get_state(self):
        return json.dumps({"learned": self.learned})
    def set_state(self, state):
        self.learned = json.loads(state)["learned"]


class ContractTests(unittest.TestCase):
    def test_user_format_hint_waiver_preserves_io_and_interface_checks(self):
        from types import SimpleNamespace
        from pilot import run_pilot as controller
        with scratch() as package:
            (package / "agents").mkdir()
            path = package / "agents/hinted.py"
            path.write_text('PROMPT = "Do not append 罪"\n',encoding="utf-8")
            with patch.object(controller,"config",return_value={"allow_task_format_hints":True}), patch.object(controller.subprocess,"run",return_value=SimpleNamespace(returncode=0)) as validate:
                controller.validate_candidate(package,"hinted")
                validate.assert_called_once()
                path.write_text('import os\nPROMPT = "Do not append 罪"\n',encoding="utf-8")
                with self.assertRaises(ValueError): controller.validate_candidate(package,"hinted")
            path.write_text('PROMPT = "Do not append 罪"\n',encoding="utf-8")
            with patch.object(controller,"config",return_value={"allow_task_format_hints":False}):
                with self.assertRaises(ValueError): controller.validate_candidate(package,"hinted")

    def test_prototype_acceptance_handles_guarded_absolute_and_relative_paths(self):
        from types import SimpleNamespace
        from pilot.proposer_contract import verify_execution
        with scratch() as package:
            tools = [SimpleNamespace(name="Read",tool_id="read",is_error=False,input={"file_path":str(package / "history/base/score/diagnostics.jsonl")},output="item evidence"),
                     SimpleNamespace(name="Bash",tool_id="a",is_error=False,input={"command":'python "'+str(package / ".prototypes/a.py")+'"'},output="ok"),
                     SimpleNamespace(name="Bash",tool_id="b",is_error=False,input={"command":"python .prototypes/b.py"},output="ok")]
            result=SimpleNamespace(raw_events=[{"type":"system","subtype":"init","tools":["Bash"]}],tool_calls=tools)
            guard=[{"tool_use_id":t.tool_id,"decision":"allow"} for t in tools]
            accepted=verify_execution(result,guard,package,[{"name":"a"},{"name":"b"}])
            self.assertTrue(accepted["passed"])
            self.assertEqual(accepted["prototype_commands_executed"],[".prototypes/a.py",".prototypes/b.py"])
            with self.assertRaises(RuntimeError):verify_execution(result,guard[:-1],package,[{"name":"a"},{"name":"b"}])

    def test_prototype_suffixes_still_require_successful_candidate_specific_execution(self):
        from types import SimpleNamespace
        from pilot.proposer_contract import verify_execution
        with scratch() as package:
            tools = [SimpleNamespace(name="Read",tool_id="read",is_error=False,input={"file_path":str(package / "history/base/score/diagnostics.jsonl")},output="item evidence"),
                     SimpleNamespace(name="Bash",tool_id="a",is_error=False,input={"command":"python .prototypes/a_proto.py"},output="ok"),
                     SimpleNamespace(name="Bash",tool_id="b",is_error=False,input={"command":"python .prototypes/b_final.py"},output="ok")]
            result=SimpleNamespace(raw_events=[{"type":"system","subtype":"init","tools":["Bash"]}],tool_calls=tools)
            guard=[{"tool_use_id":t.tool_id,"decision":"allow"} for t in tools]
            accepted=verify_execution(result,guard,package,[{"name":"a"},{"name":"b"}])
            self.assertEqual(accepted["candidate_prototypes"],{"a":[".prototypes/a_proto.py"],"b":[".prototypes/b_final.py"]})
            tools[2].is_error=True
            with self.assertRaises(RuntimeError):verify_execution(result,guard,package,[{"name":"a"},{"name":"b"}])
            tools[2].is_error=False
            tools[2].input={"command":"python .prototypes/unrelated_final.py"}
            with self.assertRaises(RuntimeError):verify_execution(result,guard,package,[{"name":"a"},{"name":"b"}])
            tools[2].input={"command":"python reports/b_final.py"}
            with self.assertRaises(RuntimeError):verify_execution(result,guard,package,[{"name":"a"},{"name":"b"}])

    def test_atomic_checkpoint_retries_transient_replace_and_retains_permanent_failure(self):
        from pilot import state
        with scratch() as folder:
            target=folder / "checkpoint.json"
            atomic_json(target,{"step":1})
            original_replace=Path.replace
            attempts=[]
            def transient_replace(source,destination):
                attempts.append(source)
                self.assertEqual(read_json(target),{"step":1})
                if len(attempts)<3:
                    raise PermissionError("synthetic sharing violation")
                return original_replace(source,destination)
            with patch.object(Path,"replace",transient_replace), patch.object(state.time,"sleep"):
                atomic_json(target,{"step":2})
            self.assertEqual(len(attempts),3)
            self.assertEqual(read_json(target),{"step":2})
            with patch.object(Path,"replace",side_effect=PermissionError("persistent denial")) as replace, patch.object(state.time,"sleep"):
                with self.assertRaises(PermissionError):atomic_json(target,{"step":3})
                self.assertEqual(replace.call_count,7)
            self.assertEqual(read_json(target),{"step":2})
            parts=list(folder.glob("checkpoint.json.part-*"))
            self.assertEqual(len(parts),1)
            self.assertEqual(read_json(parts[0]),{"step":3})

    def test_original_general_purpose_constraint_rejects_executable_crime_hints(self):
        from pilot.proposer_contract import inspect_generality, normalize_documentation
        inspect_generality('"""CJK retrieval for Chinese legal text."""\nPROMPT = "Use the exact labels observed in examples"\n')
        inspect_generality('PROMPT = "Canonical charge labels seen in training: {learned_labels}"\n')
        with self.assertRaises(ValueError): inspect_generality('PROMPT = "Do not append 罪; use this specific crime label"\n')
        original = '# LawBench note\n"""LawBench documentation"""\nPROMPT = "Use learned labels"\n'
        cleaned = normalize_documentation(original)
        self.assertNotIn('LawBench',cleaned)
        self.assertIn('PROMPT = "Use learned labels"',cleaned)
        inspect_generality(cleaned)
        with self.assertRaises(ValueError): inspect_generality(normalize_documentation('PROMPT = "LawBench answer"\n'))

    def test_readable_diagnostics_preserve_full_items_and_actual_call_details(self):
        from pilot.diagnostics import materialize
        with scratch() as directory:
            prompt = "A long memory line\n" * 6000
            trace = {"run_id":"synthetic","candidate":"base","round":0,"phase":"score","D":0,
                     "item_id":"item_1","input":"complete input","prediction":"P","target":"G","was_correct":False,
                     "call_ids":["call_1"],"metadata":{"long":prompt},"prompt_text":prompt}
            call = {"event":"api_result","call_id":"call_1","item_id":"item_1","prompt":prompt,"content":"actual output",
                    "system_prompt":"system","parameters":{"temperature":0},"model":"synthetic","raw_response":{"full":"kept"}}
            jsonl_write(directory / "score_traces.jsonl",[trace])
            jsonl_write(directory / "calls.jsonl",[call])
            before = digest(directory / "score_traces.jsonl")
            first = materialize(directory,"score")
            row = read_jsonl(directory / "diagnostics.jsonl")[0]
            for key in ["item_id","input","prediction","target","was_correct"]:
                self.assertEqual(row[key],trace[key])
            self.assertEqual((directory / row["model_calls"][0]["prompt"]).read_text(encoding="utf-8"),prompt)
            self.assertEqual(read_json(directory / row["metadata_file"]),trace["metadata"])
            self.assertLess(first["max_row_characters"],1000)
            self.assertEqual(first,materialize(directory,"score"))
            self.assertEqual(before,digest(directory / "score_traces.jsonl"))
            (directory / row["model_calls"][0]["prompt"]).write_text("changed",encoding="utf-8")
            with self.assertRaises(ValueError): materialize(directory,"score")

    def test_real_data_order_partition_prefix_and_prompt(self):
        manifest = ROOT / "external/prepared_data_v3/manifest_v3.json"
        panels = {phase: load_pilot_split(manifest, phase, 100 if phase == "feedback" else None)[0] for phase in ["train", "score", "feedback", "audit"]}
        self.assertEqual({k: len(v) for k, v in panels.items()}, {"train": 200, "score": 100, "feedback": 100, "audit": 100})
        self.assertEqual(len({r["item_id"] for rows in panels.values() for r in rows}), 500)
        old = read_jsonl(ROOT.parent / "reference_examples/text_classification/data/crime_prediction/train.jsonl")
        random.Random(42).shuffle(old)
        expected = wrap_lawbench_rows(old)
        self.assertEqual([(r["input"], r["target"]) for r in panels["train"]], [(r["input"], r["target"]) for r in expected])
        prefix, _ = load_pilot_split(manifest, "feedback", 50)
        self.assertEqual(prefix, panels["feedback"][:50])
        self.assertEqual(load_pilot_split(manifest, "feedback", 0)[0], [])
        self.assertEqual(sum(r["legacy"] for r in panels["score"]), 50)

    def test_cache_off_really_calls_twice_and_run_caches_are_separate(self):
        with scratch() as temp, patch.object(ProviderLLM, "_call_completion", side_effect=fake_response) as mock:
            with LLM(model="fake", cache_mode="off") as off:
                off("identical"); off("identical")
                self.assertEqual((off.get_usage()["api_requests"], off.get_usage()["cache_hits"]), (2, 0))
            with LLM(model="fake", cache_mode="run", cache_dir=temp / "a") as a:
                a("identical"); a("identical")
                self.assertEqual((a.get_usage()["api_requests"], a.get_usage()["cache_hits"]), (1, 1))
            with LLM(model="fake", cache_mode="run", cache_dir=temp / "b") as b:
                b("identical")
                self.assertEqual(b.get_usage()["api_requests"], 1)
            self.assertEqual(mock.call_count, 4)
            self.assertEqual(set(p.name for p in temp.iterdir()), {"a", "b"})

    def test_run_cache_cannot_fall_back_to_old_global_cache(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "explicit"):
                ProviderLLM("fake", cache_mode="run")

    def test_threaded_multi_call_item_association(self):
        class MultiCall(CountingMemory):
            def predict(self, input):
                one = self.call_llm(input + ":one")
                two = self._llm.batch([input + ":two", input + ":three"])
                return input, {"one": one, "two": two}
        with patch.object(ProviderLLM, "_call_completion", side_effect=fake_response):
            with LLM(model="fake", cache_mode="off") as llm:
                rows = [{"input": f"item{i}", "target": f"item{i}", "item_id": f"id{i}"} for i in range(32)]
                memory = MultiCall(llm)
                result = evaluate_memory(memory, rows, lambda a,b,**kw: a == b, max_workers=16, observation={"phase": "score", "run_id": "fake"})
                self.assertEqual(result["correct"], 32)
                self.assertEqual(memory.learned, 200)
                for i, r in enumerate(result["predictions"]):
                    done = [c for c in r["calls"] if c["event"] == "api_result"]
                    self.assertEqual(len(done), 3)
                    self.assertTrue(all(c["item_id"] == f"id{i}" and c["prompt"].startswith(f"item{i}:") for c in done))
                    self.assertEqual(len(r["call_ids"]), 3)
                    self.assertIn("two", r["metadata"])
                self.assertEqual(llm.get_usage()["api_requests"], 96)

    def test_no_call_prediction_does_not_inherit_last_prompt(self):
        class Conditional(CountingMemory):
            def predict(self, input):
                return (input, {}) if input == "rule_only" else (self.call_llm(input), {})
        with patch.object(ProviderLLM, "_call_completion", side_effect=fake_response):
            with LLM(model="fake", cache_mode="off") as llm:
                result = evaluate_memory(Conditional(llm), [{"input": s, "target": s} for s in ["uses_model", "rule_only"]], lambda a,b,**kw: a == b, max_workers=1)
                second = result["predictions"][1]
                self.assertEqual(second["prompt_text"], "")
                self.assertEqual(second["call_ids"], [])
                self.assertEqual(llm.get_usage()["api_requests"], 1)

    def test_controller_lock_prevents_duplicate_launch(self):
        from pilot.state import exclusive_controller
        with scratch() as temp:
            with exclusive_controller(temp / "controller.lock"):
                with self.assertRaises(RuntimeError):
                    with exclusive_controller(temp / "controller.lock"):
                        self.fail("A second controller acquired the same lock")

    def test_deferred_experiment_cannot_initialize(self):
        from pilot import run_pilot as controller
        with scratch() as temp:
            cfg = read_json(ROOT / "pilot_config.json")
            cfg["enabled_experiments"] = [1, 2, 3]
            atomic_json(temp / "pilot_config.json", cfg)
            with patch.object(controller, "ROOT", temp):
                with self.assertRaisesRegex(PermissionError, "deferred"):
                    controller.create_run("D50_c")
            self.assertFalse((temp / "runs/D50_c").exists())

    def test_proposer_relay_preserves_body_auth_and_counts_failed_attempts(self):
        import threading
        import urllib.request
        import urllib.error
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        from pilot.proposer_relay import ProposerRelay, summarize_requests
        received = []
        response_body = b'event: message_stop\ndata: {"type":"message_stop"}\n\n'
        class Upstream(BaseHTTPRequestHandler):
            def log_message(self, *args): pass
            def do_POST(self):
                body = self.rfile.read(int(self.headers["Content-Length"]))
                received.append((body, self.headers.get("Authorization")))
                self.send_response(503 if len(received) == 1 else 200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(response_body)))
                self.end_headers()
                self.wfile.write(response_body)
        server = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
        thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": .05}, daemon=True)
        thread.start()
        try:
            with scratch() as temp:
                data = b'{"model":"fixed-test-model","messages":[],"stream":true}'
                with ProposerRelay(f"http://127.0.0.1:{server.server_port}", temp / "requests.jsonl") as relay:
                    request = urllib.request.Request(relay.url + "/v1/messages", data=data, headers={"Authorization": "Bearer synthetic-test-secret", "Content-Type": "application/json"})
                    with self.assertRaises(urllib.error.HTTPError): urllib.request.urlopen(request, timeout=10)
                    with urllib.request.urlopen(request, timeout=10) as response:
                        self.assertEqual(response.read(), response_body)
                self.assertEqual(received, [(data, "Bearer synthetic-test-secret")] * 2)
                log = (temp / "requests.jsonl").read_text(encoding="utf-8")
                self.assertNotIn("synthetic-test-secret", log)
                counts = summarize_requests(read_jsonl(temp / "requests.jsonl"))
                self.assertEqual(counts["api_requests"], 2)
                self.assertEqual(counts["failed_api_requests"], 1)
                self.assertEqual(counts["repeated_request_body_attempts"], 1)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def test_api_retry_limits_and_auth_immediate_stop(self):
        class Failure(Exception):
            def __init__(self, status): self.status_code = status
        with patch("text_classification.llm.time.sleep"), patch.object(ProviderLLM, "_call_completion", side_effect=Failure(503)) as mock:
            with LLM(model="fake", cache_mode="off") as llm:
                with self.assertRaises(Failure): llm("x")
                self.assertEqual(mock.call_count, 4)
                self.assertEqual(llm.get_usage()["failed_requests"], 4)
        with patch.object(ProviderLLM, "_call_completion", side_effect=Failure(401)) as mock:
            with LLM(model="fake", cache_mode="off") as llm:
                with self.assertRaises(Failure): llm("x")
                self.assertEqual(mock.call_count, 1)

    def test_evaluation_failure_preserves_finished_items_and_cancels_queue(self):
        finished, saved = [], []
        class Failing(CountingMemory):
            def predict(self, input):
                if input == "bad":
                    time.sleep(.01)
                    raise RuntimeError("synthetic auth failure")
                time.sleep(.04)
                finished.append(input)
                return input, {}
        rows = [{"input": "bad" if i == 0 else str(i), "target": str(i), "item_id": str(i)} for i in range(100)]
        with self.assertRaises(RuntimeError):
            evaluate_memory(Failing(lambda x: x), rows, lambda a,b,**kw: a == b, max_workers=4,
                            on_prediction=lambda i,r: saved.append(r["input"]))
        self.assertEqual(set(finished), set(saved))
        self.assertLess(len(finished), 100)

    def test_feedback_loads_state_never_learns_and_resume_does_not_call(self):
        from pilot.worker import execute
        with scratch() as temp:
            memory_file = temp / "memory.json"
            memory_file.write_text('{"learned":200}', encoding="utf-8")
            created = []
            def factory(path, llm):
                memory = CountingMemory(llm)
                created.append(memory)
                return memory
            calls = []
            def fake(prompt): calls.append(prompt); return "synthetic_test_only"
            spec = {"run_id": "fake", "candidate": "confusion_disambiguation_memory", "round": 1, "D": 50,
                    "phase": "feedback", "package": str(ROOT / "engine/text_classification"), "output": str(temp / "feedback"),
                    "memory": str(memory_file), "cache_mode": "off", "cache_dir": None}
            with patch("text_classification.inner_loop.load_memory_system", side_effect=factory):
                result = execute(spec, llm_override=fake)
                self.assertEqual(len(calls), 50)
                self.assertEqual(created[0].learned, 200)
                self.assertFalse(result["persistent_prediction_state_changed"])
                self.assertEqual(len(read_jsonl(temp / "feedback/feedback_traces.jsonl")), 50)
                execute(spec, llm_override=fake)
                self.assertEqual(len(calls), 50)
                zero_spec = {**spec, "D": 0, "output": str(temp / "zero")}
                zero = execute(zero_spec, llm_override=fake)
                self.assertEqual(zero["total"], 0)
                self.assertEqual(len(calls), 50)
                (temp / "feedback/feedback_traces.jsonl").write_text("tampered", encoding="utf-8")
                with self.assertRaises(ValueError): execute(spec, llm_override=fake)

    def test_train_resume_from_last_completed_step(self):
        from pilot.worker import execute
        with scratch() as temp:
            calls = []
            fail_at = [7]
            def fake(prompt):
                if len(calls) == fail_at[0]:
                    fail_at[0] = -1
                    raise RuntimeError("synthetic interruption before request")
                calls.append(prompt)
                return "synthetic_test_only"
            spec = {"run_id": "fake", "candidate": "confusion_disambiguation_memory", "round": 1, "D": 0,
                    "phase": "train", "package": str(ROOT / "engine/text_classification"), "output": str(temp / "train"),
                    "memory": None, "cache_mode": "off", "cache_dir": None}
            with patch("text_classification.inner_loop.load_memory_system", side_effect=lambda path,llm: CountingMemory(llm)), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(RuntimeError): execute(spec, llm_override=fake)
                self.assertEqual(read_json(temp / "train/training_checkpoint.json")["next_step"], 7)
                result = execute(spec, llm_override=fake)
                self.assertEqual(result["total"], 200)
                self.assertEqual(len(calls), 200)
                self.assertEqual(len(read_jsonl(temp / "train/train_traces.jsonl")), 200)
                execute(spec, llm_override=fake)
                self.assertEqual(len(calls), 200)

    def test_test_files_other_runs_and_frozen_code_denied(self):
        from pilot.access_guard import check_tool
        with scratch() as temp:
            root = temp / "run"
            root.mkdir()
            for event in [
                {"tool_name": "Read", "tool_input": {"file_path": str(temp / "audit.json")}},
                {"tool_name": "Grep", "tool_input": {"path": "../other_run"}},
                {"tool_name": "Write", "tool_input": {"file_path": "config.yaml"}},
                {"tool_name": "Edit", "tool_input": {"file_path": "agents/base.py"}},
                {"tool_name": "Bash", "tool_input": {"command": "python x.py; env"}},
                {"tool_name": "WebSearch", "tool_input": {"query": "answers"}},
            ]:
                with self.assertRaises(ValueError): check_tool(root, {"agents/base.py"}, event)
            check_tool(root, set(), {"tool_name": "Read", "tool_input": {"file_path": "history/score_traces.jsonl"}})
            check_tool(root, {"agents/base.py"}, {"tool_name": "Write", "tool_input": {"file_path": "agents/new.py"}})

    def test_budget_ledger_idempotent_and_failed_requests_retained(self):
        from pilot.state import BudgetExceeded, Ledger
        with scratch() as temp:
            ledger = Ledger(temp / "ledger.jsonl", 1)
            start = {"event": "api_started", "attempt_id": "a", "event_id": "a:start", "reserved_usd": .25}
            done = {"event": "api_result", "attempt_id": "a", "event_id": "a:done", "cost": .1}
            for row in [start, done, done]: ledger.add(row)
            self.assertAlmostEqual(ledger.summary()["known_or_estimated_usd"], .1)
            self.assertEqual(len(read_jsonl(temp / "ledger.jsonl")), 2)
            ledger.add({"event": "api_started", "attempt_id": "b", "event_id": "b:start", "reserved_usd": .25})
            ledger.add({"event": "api_failed", "attempt_id": "b", "event_id": "b:fail", "cost": None})
            self.assertAlmostEqual(ledger.summary()["reserved_or_unknown_usd"], .25)
            with self.assertRaises(BudgetExceeded):
                ledger.add({"event": "api_started", "attempt_id": "c", "event_id": "c:start", "reserved_usd": .8})

    def test_credit_rejections_release_unspent_budget_and_preflight_blocks(self):
        from pilot.state import Ledger
        from pilot import billing
        with scratch() as temp:
            ledger = Ledger(temp / "ledger.jsonl", 1)
            ledger.add({"event": "api_started", "attempt_id": "a", "event_id": "a:start", "reserved_usd": .25})
            ledger.add({"event": "api_failed", "attempt_id": "a", "event_id": "a:failed", "http_status": 402, "cost": None})
            self.assertEqual(ledger.summary()["reserved_or_unknown_usd"], 0)
            self.assertEqual(len(read_jsonl(temp / "ledger.jsonl")), 2)
            atomic_json(temp / "pilot_config.json", {"credentials_file": "unused.env"})
            def response(payload):
                obj = io.BytesIO(json.dumps({"data": payload}).encode())
                obj.status = 200
                return obj
            with patch.object(billing, "ROOT", temp), patch("dotenv.dotenv_values", return_value={"OPENROUTER_API_KEY": "synthetic-key"}), patch("urllib.request.urlopen", side_effect=[response({"limit_remaining": None}), response({"total_credits": 1, "total_usage": 1.2})]):
                with self.assertRaisesRegex(RuntimeError, "available credits"):
                    billing.ensure_solver_credits()
            self.assertLess(read_json(temp / "external/openrouter_billing_preflight.json")["available_account_credits_usd"], 0)

    def test_tie_rule_and_predefined_signal_thresholds(self):
        from pilot.run_pilot import select_earliest
        from pilot.analyze import signal
        self.assertEqual(select_earliest([{"candidate": "H0", "correct": 60}, {"candidate": "new", "correct": 60}])["candidate"], "H0")
        self.assertEqual(signal([3, 4], 3), "positive")
        self.assertEqual(signal([-3, -4], 3), "negative")
        self.assertEqual(signal([-1, 7], 3), "unresolved")
        self.assertEqual(signal([1, 2], 3), "unresolved")

    def test_full_two_round_state_machine_cost_snapshot_and_prediction_gate(self):
        from pilot import run_pilot as controller
        from pilot import analyze
        with scratch() as temp:
            fixture_config = read_json(ROOT / "pilot_config.json")
            fixture_config["enabled_experiments"] = [1,2,3,4]
            atomic_json(temp / "pilot_config.json", fixture_config)
            atomic_json(temp / "analysis/noise_reference.json", {"S0_pp": 40, "J_pp": 1, "delta_pp": 3})
            stages, generation = [], []
            def fake_create(run):
                if run == "D50_c": analyze.verify_prediction()
                package = controller.package_for(run)
                (package / "agents").mkdir(parents=True, exist_ok=True)
                (package / "agents" / (controller.BASE + ".py")).write_text("# fixed base", encoding="utf-8")
                target = package / "history" / controller.BASE / "train"
                target.mkdir(parents=True, exist_ok=True)
                (target / "memory.json").write_text("{}", encoding="utf-8")
                controller.control_for(run).mkdir(parents=True, exist_ok=True)
                if not (package / "evolution_summary.jsonl").exists():
                    jsonl_write(package / "evolution_summary.jsonl", [])
                atomic_json(controller.control_for(run) / "initialized.json", {"test_only": True})
                return package
            def fake_stage(run, name, T, phase):
                output = controller.stage_output(run, name, phase)
                existing = output / "result.json"
                if existing.exists(): return read_json(existing)
                output.mkdir(parents=True, exist_ok=True)
                D = controller.config()["runs"][run]
                total = 200 if phase == "train" else D if phase == "feedback" else 100
                stages.append({"run": run, "candidate": name, "T": T, "phase": phase, "total": total, "cost": total * .001})
                correct = 40 if name == controller.BASE else 40 + T + D // 50
                correct = min(correct, total)
                result = {"correct": correct, "total": total, "accuracy": correct / total if total else 0,
                          "phase": phase, "run_id": run, "candidate": name, "round": T, "D": D,
                          "runtime_seconds": 0, "runtime_seconds_this_attempt": 0, "cache_mode": "run",
                          "usage": {"cache_hits": 0, "api_requests": 0}, "estimated_cost_usd": total * .001}
                if phase == "audit": result["hidden_canary"] = "AUDIT_TEST_ONLY_CANARY"
                atomic_json(existing, result)
                if phase == "train": (output / "memory.json").write_text("{}", encoding="utf-8")
                if phase in {"score", "audit"}:
                    jsonl_write(output / f"{phase}_traces.jsonl", [{"item_id": f"{phase}_{i}", "was_correct": i < correct, "legacy": i < 50} for i in range(total)])
                return result
            def fake_propose(run, T):
                saved = controller.control_for(run) / f"proposer_round{T}/pending_eval.json"
                if saved.exists(): return read_json(saved)["candidates"]
                history = controller.package_for(run) / "history"
                required = [controller.BASE] if T == 1 else [f"candidate_1_{slot}" for slot in "ab"]
                self.assertTrue(all((history / name / "feedback/result.json").exists() for name in required))
                for p in controller.package_for(run).rglob("*.json*"):
                    self.assertNotIn("AUDIT_TEST_ONLY_CANARY", p.read_text(encoding="utf-8"))
                generation.append((run, T))
                candidates = []
                for slot in "ab":
                    name = f"candidate_{T}_{slot}"
                    source = controller.package_for(run) / "agents" / f"{name}.py"
                    source.write_text(f"# new {T} {slot}", encoding="utf-8")
                    candidates.append({"name": name, "file": f"agents/{name}.py", "slot": slot.upper(), "code_hash": digest(source)})
                atomic_json(saved, {"iteration": T, "candidates": candidates})
                return candidates
            def snapshot(run):
                relevant = [r for r in stages if r["run"] == run and r["phase"] in controller.MAIN_PHASES]
                return {"cost_usd": sum(r["cost"] for r in relevant) + sum(r == run for r,t in generation) * .1, "unknown_failed_attempts": 0}
            with patch.object(controller, "ROOT", temp), patch.object(analyze, "ROOT", temp), patch.object(analyze, "ANALYSIS", temp / "analysis"), patch.object(controller, "create_run", side_effect=fake_create), patch.object(controller, "run_stage", side_effect=fake_stage), patch.object(controller, "propose", side_effect=fake_propose), patch.object(controller, "main_cost_snapshot", side_effect=snapshot):
                with self.assertRaises(FileNotFoundError): controller.round_run("D50_c", 1)
                saved_round1 = {}
                for run in controller.CORE_RUNS:
                    controller.round_run(run, 1)
                    cp = controller.control_for(run) / "round1_checkpoint.json"
                    saved_round1[run] = digest(cp)
                    D = controller.config()["runs"][run]
                    self.assertEqual(sum(r["total"] for r in stages if r["run"] == run and r["phase"] == "feedback"), D)
                    self.assertAlmostEqual(read_json(cp)["cost"]["cost_usd"], .8 + D * .001)
                for run in controller.CORE_RUNS:
                    controller.round_run(run, 2)
                    self.assertEqual(digest(controller.control_for(run) / "round1_checkpoint.json"), saved_round1[run])
                    D = controller.config()["runs"][run]
                    self.assertEqual(sum(r["total"] for r in stages if r["run"] == run and r["phase"] == "feedback"), 3 * D)
                    self.assertFalse(any(r["T"] == 2 and r["phase"] == "feedback" for r in stages))
                prediction = analyze.freeze_prediction()
                seal = digest(temp / "analysis/prediction_before_run.json")
                for T in [1,2]: controller.round_run("D50_c", T)
                self.assertEqual(len(generation), 10)
                self.assertEqual(sum(r["total"] for r in stages if r["phase"] == "train"), 4000)
                self.assertEqual(sum(r["total"] for r in stages if r["phase"] == "feedback"), 750)
                self.assertEqual(sum(r["total"] for r in stages if r["phase"] == "audit"), 2000)
                self.assertEqual(sum(r["total"] for r in stages if r["phase"] == "score"), 2500)
                analyze.verify_prediction()
                self.assertEqual(digest(temp / "analysis/prediction_before_run.json"), seal)
                # Run the real report/analysis path on clearly synthetic, disposable data.
                for label, correct in [("R4B", 40), ("R7A", 42), ("R12B", 43)]:
                    for rep in [1,2]:
                        for phase in ["score", "audit"]:
                            folder = temp / "external/noise" / label / str(rep) / phase
                            result = {"run_id": f"noise_{label}_{rep}", "candidate": label, "round": 0, "D": 0,
                                      "phase": phase, "correct": correct, "total": 100, "accuracy": correct / 100,
                                      "estimated_cost_usd": 0, "runtime_seconds_this_attempt": 0, "cache_mode": "off",
                                      "usage": {"api_requests": 0, "cache_hits": 0}}
                            atomic_json(folder / "result.json", result)
                            jsonl_write(folder / f"{phase}_traces.jsonl", [{"item_id": f"{phase}_{i}", "was_correct": i < correct} for i in range(100)])
                initial_reference = read_json(temp / "analysis/noise_reference.json")
                initial_reference["input_hashes"] = {p.relative_to(temp).as_posix(): digest(p) for p in (temp / "external/noise").rglob("result.json")}
                atomic_json(temp / "analysis/noise_reference.json", initial_reference)
                # Endpoints here select new candidates, so prediction inputs do not use S0's file hash.
                with patch.object(analyze.subprocess, "run") as plot:
                    report = analyze.analyze_all()
                    self.assertEqual(report["valid_new_candidates"], 20)
                    self.assertEqual(report["proposer_sessions"], 10)
                    self.assertTrue((temp / "analysis/paired_changes.csv").exists())
                    plot.assert_called_once()
                    core_report = analyze.analyze_all(include_holdout=False)
                    self.assertEqual(core_report["valid_new_candidates"], 16)
                    self.assertEqual(core_report["proposer_sessions"], 8)
                    self.assertEqual(core_report["prediction"]["prediction_status"], "not_run_by_user_request")
                    self.assertEqual(core_report["deferred_experiments"], [4])
                before = len(stages), len(generation)
                controller.round_run("D100_a", 2)
                self.assertEqual((len(stages), len(generation)), before)
                changed = {**prediction, "D50_S_prediction_pp": 99}
                atomic_json(temp / "analysis/prediction_before_run.json", changed)
                with self.assertRaises(ValueError): analyze.verify_prediction()


def run_verification():
    # The original package tests use TemporaryDirectory(mode=0700). On this Windows
    # restricted token, inherit the test workspace ACL without changing production code.
    temp_root = ROOT / ".verification_tmp"
    temp_root.mkdir(exist_ok=True)
    original_mkdir = os.mkdir
    def inherited_mkdir(path, mode=0o777, *, dir_fd=None):
        if dir_fd is None: return original_mkdir(path, 0o777)
        return original_mkdir(path, 0o777, dir_fd=dir_fd)
    previous_temp = tempfile.tempdir
    tempfile.tempdir = str(temp_root)
    output = io.StringIO()
    sys.path.insert(0, str(ROOT / "protocol"))
    suite = unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromTestCase(ContractTests),
                               unittest.defaultTestLoader.discover(str(ROOT / "protocol"), pattern="test_prepare_data.py")])
    try:
        with patch("os.mkdir", inherited_mkdir), patch("text_classification.llm.litellm_completion", side_effect=AssertionError("Real model calls are forbidden in offline tests")):
            result = unittest.TextTestRunner(stream=output, verbosity=2).run(suite)
    finally:
        tempfile.tempdir = previous_temp
        set_event_callback(None)
    report = {"passed": result.wasSuccessful(), "tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors), "model_api_requests": 0,
              "scope": "offline synthetic execution and actual dataset integrity; not provider or paid experiment validation",
              "source_hashes": {p.relative_to(ROOT).as_posix(): digest(p) for folder in [ROOT / "pilot", ROOT / "engine"] for p in folder.rglob("*.py")}}
    (ROOT / "offline_test_results.txt").write_text(output.getvalue(), encoding="utf-8")
    atomic_json(ROOT / "verification.json", report)
    print(output.getvalue())
    return report


if __name__ == "__main__":
    raise SystemExit(not run_verification()["passed"])
