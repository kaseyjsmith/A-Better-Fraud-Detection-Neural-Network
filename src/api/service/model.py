from src.models.nn.architectures.deep import DeepFraudNN

from pathlib import Path

proj_root = "/home/ksmith/birds/neural_networks/fraud_detection"


class PredicitonModel:
    def __init__(self):
        self.model = self.load_model()

    def load_model(
        self,
        checkpoint_path=f"{proj_root}/lightning_logs/deep_e5_lr0.001_b512/version_0/checkpoints/epoch=4-step=2090.ckpt",
    ):
        # load the model
        return DeepFraudNN.load_from_checkpoint(
            checkpoint_path,
            pos_weight=258.0,  # see /experiements/hyperparameter_configs.json for details
            lr=0.001,
            threshold=0.9,
        )
