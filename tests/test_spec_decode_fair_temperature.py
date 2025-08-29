import inspect
from training.utils import measure_generate_walltime


def test_measure_generate_walltime_signature_has_temperature():
    sig = inspect.signature(measure_generate_walltime)
    assert "temperature" in sig.parameters
