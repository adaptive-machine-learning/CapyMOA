import sys

import pytest


def test_old_events_path_warns_and_reexports():
    for name in list(sys.modules):
        if name.startswith("capymoa.ocl.evaluation"):
            del sys.modules[name]

    with pytest.warns(DeprecationWarning):
        from capymoa.ocl.evaluation.events import TrainTaskBegin

    from capymoa.ocl.evaluate.events import TrainTaskBegin as New

    assert TrainTaskBegin is New
