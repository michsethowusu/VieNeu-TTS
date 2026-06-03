import os
import sys

# Point to Twi-specific config
os.environ.setdefault("VIENEU_CONFIG",  os.path.join(os.path.dirname(__file__), "config_twi.yaml"))
os.environ.setdefault("VIENEU_LANG",    "twi")
os.environ.setdefault("VIENEU_EMOTION", "none")

# Locate apps/ — works both in Docker (PYTHONPATH=/opt/vieneutts) and local dev
_here = os.path.dirname(os.path.abspath(__file__))
_candidates = [
    "/opt/vieneutts",                                    # Docker
    os.path.dirname(os.path.dirname(_here)),             # local: space_twi/../../
]
for _root in _candidates:
    if os.path.exists(os.path.join(_root, "apps")):
        sys.path.insert(0, _root)
        break

from apps.gradio_main import demo  # noqa: E402

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
