"""
Tests for solvation properties. This tests the mechanics of the workflow
"""
import tempfile

import pytest

from openff.evaluator.datasets import PhysicalPropertyDataSet
from openff.evaluator.properties import SolvationFreeEnergy
from openff.evaluator.attributes.attributes import UndefinedAttribute
from openff.evaluator.workflow import Workflow
from openff.evaluator.utils.utils import get_data_filename, temporarily_change_directory


def modify_workflow_schema(workflow_schema):
    """
    Modify a SolvationFreeEnergy schema to run faster for testing.
    """
    new_workflow_schema = []
    for schema in workflow_schema.protocol_schemas:
        if schema.id == "equilibration_simulation":
            protocol = schema.to_protocol()
            protocol.steps_per_iteration = 200
            schema = protocol.schema

        if schema.id == "conditional_group":
            protocol = schema.to_protocol()
            yank_protocol = protocol.protocols["run_solvation_yank"]
            yank_protocol.number_of_equilibration_iterations = 1
            yank_protocol.number_of_iterations = 10
            yank_protocol.checkpoint_frequency = 1
            yank_protocol.steps_per_iteration = 100
            yank_protocol.electrostatic_lambdas_1 = [1.0, 0.5, 0.0]
            yank_protocol.steric_lambdas_1 = [1.0, 0.5, 0.0]
            yank_protocol.electrostatic_lambdas_2 = [1.0, 0.5, 0.0]
            yank_protocol.steric_lambdas_2 = [1.0, 0.5, 0.0]
            protocol.max_iterations = 1
            schema = protocol.schema

        new_workflow_schema.append(schema)

    workflow_schema.protocol_schemas = new_workflow_schema
    workflow_schema.replace_protocol_types(
        {"BaseBuildSystem": "BuildSmirnoffSystem"},
    )
    return workflow_schema


class TestSolvationFreeEnergy:
    @pytest.mark.parametrize("forcefield_name", [
        "sage-with-tip3p.json",         # no virtual sites
        "sage-with-opc.json",           # solvent with virtual sites
        # not supported by openmmtools yet
        # "test-vsites-halogens.json",    # solute with virtual sites
    ])
    def test_run_with_vsite_solvent(self, forcefield_name):
        # load force field
        forcefield_json = get_data_filename(f"test/forcefields/{forcefield_name}")

        # load dataset
        dataset_path = get_data_filename("test/datasets/single-sfe-dataset.json")
        dataset = PhysicalPropertyDataSet.from_json(dataset_path)

        # generate metadata
        metadata = Workflow.generate_default_metadata(
            dataset.properties[0],
            forcefield_json
        )

        default_schema = SolvationFreeEnergy.default_simulation_schema(n_molecules=250)
        workflow_schema = modify_workflow_schema(default_schema.workflow_schema)

        workflow = Workflow.from_schema(workflow_schema, metadata=metadata)

        with tempfile.TemporaryDirectory() as directory:
            with temporarily_change_directory(directory):
                result = workflow.execute()
                assert not isinstance(result.value, UndefinedAttribute)
                assert not result.exceptions
                                