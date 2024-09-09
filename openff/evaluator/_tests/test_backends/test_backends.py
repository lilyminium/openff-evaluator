from openff.evaluator.backends.backends import QueueWorkerResources

import pytest

class TestQueueWorkerResources:

    @pytest.mark.parametrize("walltime", ["1:00", "01:00", "1:00:00"])
    def test_valid_walltime(self, walltime):
        QueueWorkerResources(wallclock_time_limit=walltime)
    

    @pytest.mark.parametrize("walltime", ["1:0", ""])
    def test_invalid_walltime(self, walltime):
        with pytest.raises(AssertionError):
            QueueWorkerResources(wallclock_time_limit=walltime)
