"""Identical no-API acceptance for each generated candidate."""
import json
import sys
from pathlib import Path

package, name = Path(sys.argv[1]).resolve(), sys.argv[2]
sys.path.insert(0, str(package.parent))
from text_classification.inner_loop import load_memory_system

calls = []
def fake(prompt):
    calls.append(prompt)
    return '{"reasoning":"offline interface validation", "final_answer":"label_a"}'

memory = load_memory_system(f"agents/{name}.py", fake)
prediction, metadata = memory.predict("An offline validation example")
assert isinstance(prediction, str) and isinstance(metadata, dict)
memory.learn_from_batch([{"input": "An offline validation example", "raw_question": "An offline validation example", "prediction": prediction, "ground_truth": "label_a", "was_correct": prediction == "label_a", "metadata": metadata}])
state = memory.get_state()
json.loads(state)
clone = load_memory_system(f"agents/{name}.py", fake)
clone.set_state(state)
prediction, metadata = clone.predict("Another offline validation example")
assert isinstance(prediction, str) and isinstance(metadata, dict)
print(json.dumps({"valid": True, "fake_calls": len(calls), "real_api_calls": 0}))
