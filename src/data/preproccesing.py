from pathlib import Path
from src.config.load_config import load_config

def preprocess():
    paths = load_config()['paths']    
    raw_data_path = Path(paths['raw_data'])

