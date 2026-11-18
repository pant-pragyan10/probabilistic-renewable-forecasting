from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(not (ROOT / "results" / "predictions" / "solar_neural_net_val.parquet").exists(),
                                reason="needs saved predictions (run the pipeline first)")


def test_app_renders_without_errors_for_both_tracks():
    at = AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=120).run()
    assert not at.exception
    at.sidebar.radio[1].set_value("wind").run()
    assert not at.exception
    assert len(at.tabs) == 6