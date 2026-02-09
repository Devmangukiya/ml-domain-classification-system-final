import json
import pytest
import utils

from components import train


@pytest.mark.training
def test_train_model(dataset_loc):
    experiment_name = utils.generate_experiment_name(prefix="test_train")

    train_loop_config = {
        "dropout_p": 0.3,
        "lr": 1e-4,          # faster learning for small test
        "lr_factor": 0.8,
        "lr_patience": 1,
    }

    result = train.train_model(
        experiment_name=experiment_name,
        dataset_loc=dataset_loc,
        train_loop_config=json.dumps(train_loop_config),
        num_workers=1,        # keep minimal for tests
        cpu_per_worker=1,
        gpu_per_worker=0,
        num_epochs=2,         # safe for CI
        num_samples=128,
        batch_size=32,
        results_fp=None,
    )

    utils.delete_experiment(experiment_name=experiment_name)

    train_loss_list = result.metrics_dataframe["train_loss"].tolist()

    assert len(train_loss_list) >= 2
    assert train_loss_list[-1] <= train_loss_list[0]
