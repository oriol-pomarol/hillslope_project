from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class main:
    model_training: str = 'all' # 'none', 'rf', 'nn' or 'all'.
    model_evaluation: str = 'all'   # 'none', 'train', 'test' or 'all'
    fwd_sim: list = []  # [0,1,2,...,'val_data_sin','val_data_lin']
    plots: list = []    # ['surface', 'colormesh', 'tipping']

@dataclass(frozen=True)
class data_preparation:

    # DATA SOURCE
    data_source: str = 'minimal'   # 'detailed' or 'minimal'

    # DATA LOADING (if data_source == 'detailed')
    data_folder: str = 'detailed_jp'   # Name of the folder where the data is located

    # DATA GENERATION (if data_source == 'minimal')
    n_sim: int = 2000   # Number of simulations to generate
    n_years: int = 10000    # Number of years per simulation
    dt: float = 0.5   # Time step in years
    prob_new_state: float = 0.02    # Probability of jumping to a new system state
    prob_new_g: float = 0.002   # Probability of jumping to a new g value

    # DATA FORMATTING
    mask_zeroes: bool = True    # Mask zeroes in the data
    test_size: float = 0.2
    val_size: float = 0.1
    drop_size: float = 0.0

@dataclass(frozen=True)
class paths:
    # DATA PATHS
    raw_data: Path = Path('../data/raw')
    processed_data: Path = Path('../data/processed')
    temp_data: Path = Path('../data/temp')

    # RESULTS PATHS
    figures: Path = Path('../results/figures')
    outputs: Path = Path('../results/outputs')
    models: Path = Path('../results/models')
