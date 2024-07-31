"""
A collection of openff-evaluator compute backends which use parsl as the distribution engine.
"""
import abc
import importlib
import logging
import multiprocessing
import os
import platform
import shutil
import traceback

import dask
import math
import parsl
from distributed import get_worker

from openff import evaluator
from openff.evaluator.backends import (
    CalculationBackend,
    ComputeResources,
    QueueWorkerResources,
)
from openff.evaluator.backends.dask import _Multiprocessor, BaseDaskJobQueueBackend
from openff.units import unit
from openff.evaluator.utils.utils import timestamp_formatter

logger = logging.getLogger(__name__)


class BaseParslBackend(CalculationBackend, abc.ABC):

    def __init__(self, number_of_workers=1, resources_per_worker=ComputeResources()):
        super().__init__(number_of_workers, resources_per_worker)

        # initialize empty defaults
        self._jobids = []
        self._executor = None
        self._parsl_config = None
        self._dataflow_kernel = None


    def _get_base_provider_kwargs(self):
        return {
            "min_blocks": self._minimum_number_of_workers,
            "max_blocks": self._maximum_number_of_workers,
            "worker_init": self._get_worker_init(),
        }
    
    def _get_worker_init(self):
        return "; ".join(self._setup_script_commands)

    def _launch_provider(self):
        raise NotImplementedError
    
    @classmethod
    def _get_provider_class(cls):
        raise NotImplementedError
    
    @classmethod
    def _get_executor_class(cls):
        raise NotImplementedError
    
    def _get_executor_kwargs(self):
        raise NotImplementedError

    def _get_provider_kwargs(self):
        raise NotImplementedError


class ParslClusterBackend(BaseParslBackend):

    _wrapped_function = BaseDaskJobQueueBackend._wrapped_function

    def __init__(
        self,
        minimum_number_of_workers=1,
        maximum_number_of_workers=1,
        resources_per_worker=QueueWorkerResources(),
        queue_name: str = "default",
        account_name: str = None,
        setup_script_commands=None,
        extra_script_options=None,
    ):
        """Constructs a new BaseParslBackend object.

        Parameters
        ----------
        setup_script_commands: list of str, optional
            A list of commands to run before each task is executed.
        """

        super().__init__(minimum_number_of_workers, resources_per_worker)

        assert isinstance(resources_per_worker, QueueWorkerResources)

        if setup_script_commands is None:
            setup_script_commands = []
        if extra_script_options is None:
            extra_script_options = []

        self._setup_script_commands = list(setup_script_commands)
        self._extra_script_options = list(extra_script_options)

        
        self._queue_name = queue_name
        self._account_name = account_name
        self._minimum_number_of_workers = minimum_number_of_workers
        self._maximum_number_of_workers = maximum_number_of_workers

    @classmethod
    def _get_executor_class(cls):
        from parsl.executors import HighThroughputExecutor
        return HighThroughputExecutor
    
    def _get_executor_kwargs(self):
        memory = self._resources_per_worker.memory_per_thread
        kwargs = {
            "cores_per_worker": self._resources_per_worker.number_of_threads,
            "mem_per_worker": memory.m_as(unit.gigabytes),
        }
        return kwargs
    
    def _launch_executor(self):
        from parsl.channels import LocalChannel
        lchannel = LocalChannel()

        executor_class = self._get_executor_class()
        provider_class = self._get_provider_class()
        executor_kwargs = self._get_executor_kwargs()
        provider_kwargs = self._get_provider_kwargs()
        self._executor = executor_class(
            provider=provider_class(
                channel=lchannel,
                **provider_kwargs
            ),
            **executor_kwargs
        )

    def start(self):
        self._launch_executor()
        self._executor.start()
        self._executor.initialize_scaling()

    def stop(self):
        self._executor.scale_in(self._maximum_number_of_workers)
        self._executor.shutdown()

    def _get_provider_kwargs(self):
        memory = (
            self._resources_per_worker.memory_per_thread
            * self._resources_per_worker.number_of_threads
        ).m_as(unit.gigabytes)

        time_limit = self._resources_per_worker.wallclock_time_limit,
        # count how many colons are in time_limit
        if time_limit.count(":") == 1:
            time_limit = time_limit + ":00"

        kwargs = self._get_base_provider_kwargs()
        kwargs.update({
            "walltime": time_limit,
            "nodes_per_block": 1,
            "cores_per_node": self._resources_per_worker.number_of_threads,
            "mem_per_node": math.ceil(memory),
            "init_blocks": 1,
        })
        return kwargs
    
    def submit_task(self, function, *args, **kwargs):
        self._executor.submit(
            self._wrapped_function,
            {}, # resource_specification
            function,
            *args,
            **kwargs
        )


class ParslSLURMBackend(ParslClusterBackend):
    @classmethod
    def _get_provider_class(cls):
        return parsl.providers.SlurmProvider
    
    def _get_provider_kwargs(self):
        kwargs = super()._get_provider_kwargs()
        kwargs["partition"] = self._queue_name
        kwargs["account"] = self._account_name
        return kwargs



