"""
Configuration management using Pydantic + YAML.
Supports environment variable overrides and CLI overrides.
"""
import os
from pathlib import Path
from typing import List, Optional
import yaml
from pydantic import BaseModel, Field, validator


class AppConfig(BaseModel):
    name: str = "XAI-IDS Pro"
    version: str = "1.0.0"
    description: str = "Explainable AI Intrusion Detection System"
    log_level: str = "INFO"
    log_file: str = "logs/xai_ids.log"


class FeatureSelectionConfig(BaseModel):
    enabled: bool = True
    variance_threshold: float = 0.01
    correlation_threshold: float = 0.95


class DataConfig(BaseModel):
    path: str = "data/"
    file_format: str = "parquet"
    label_column: str = "Label"
    benign_label: str = "benign"
    sample_size: Optional[int] = 50000
    test_split: float = 0.2
    random_state: int = 42
    feature_selection: FeatureSelectionConfig = FeatureSelectionConfig()


class AutoencoderConfig(BaseModel):
    encoder_dims: List[int] = [128, 64, 32]
    latent_dim: int = 16
    decoder_dims: List[int] = [32, 64, 128]
    activation: str = "relu"
    dropout_rate: float = 0.2
    use_batch_norm: bool = True
    epochs: int = 50
    batch_size: int = 256
    learning_rate: float = 0.001
    early_stopping_patience: int = 10
    reduce_lr_patience: int = 5
    validation_split: float = 0.1


class ThresholdConfig(BaseModel):
    strategy: str = "percentile"
    percentile: float = 95
    iqr_multiplier: float = 1.5
    zscore_threshold: float = 3.0


class EnsembleConfig(BaseModel):
    use_isolation_forest: bool = True
    isolation_contamination: float = 0.05
    ensemble_weight_ae: float = 0.7
    ensemble_weight_if: float = 0.3


class PersistenceConfig(BaseModel):
    model_dir: str = "models/"
    save_format: str = "keras"


class ModelConfig(BaseModel):
    autoencoder: AutoencoderConfig = AutoencoderConfig()
    threshold: ThresholdConfig = ThresholdConfig()
    ensemble: EnsembleConfig = EnsembleConfig()
    persistence: PersistenceConfig = PersistenceConfig()


class ShapConfig(BaseModel):
    enabled: bool = True
    num_background_samples: int = 200
    num_explain_samples: int = 100
    plot_top_features: int = 20


class LimeConfig(BaseModel):
    enabled: bool = True
    num_features: int = 15
    num_samples: int = 1000
    kernel_width: float = 0.75


class ExplainabilityConfig(BaseModel):
    shap: ShapConfig = ShapConfig()
    lime: LimeConfig = LimeConfig()


class EvaluationConfig(BaseModel):
    metrics: List[str] = ["precision", "recall", "f1", "roc_auc", "confusion_matrix"]
    plot_roc: bool = True
    plot_precision_recall: bool = True
    plot_confusion_matrix: bool = True
    report_dir: str = "reports/"


class ApiConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    secret_key: str = "CHANGE_ME_IN_PRODUCTION"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60
    rate_limit: int = 100
    cors_origins: List[str] = ["*"]


class DashboardConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8501
    title: str = "XAI-IDS Pro Dashboard"
    theme: str = "dark"


class Config(BaseModel):
    app: AppConfig = AppConfig()
    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    explainability: ExplainabilityConfig = ExplainabilityConfig()
    evaluation: EvaluationConfig = EvaluationConfig()
    api: ApiConfig = ApiConfig()
    dashboard: DashboardConfig = DashboardConfig()

    @classmethod
    def from_yaml(cls, path: str = "config.yaml") -> "Config":
        """Load config from a YAML file, then apply env variable overrides."""
        config_path = Path(path)
        if not config_path.exists():
            return cls()
        with open(config_path) as f:
            data = yaml.safe_load(f)
        cfg = cls(**data)

        # Environment variable overrides (e.g., XAI_API__SECRET_KEY)
        secret = os.getenv("XAI_API__SECRET_KEY")
        if secret:
            cfg.api.secret_key = secret
        log_level = os.getenv("XAI_APP__LOG_LEVEL")
        if log_level:
            cfg.app.log_level = log_level.upper()
        return cfg

    def to_yaml(self, path: str = "config.yaml") -> None:
        with open(path, "w") as f:
            yaml.dump(self.dict(), f, default_flow_style=False)
