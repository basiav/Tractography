from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    penalizer: float = 0.01
    duration_col: str = 'OS (corrected)'
    event_col: str = 'status'
    n_components: int = 100