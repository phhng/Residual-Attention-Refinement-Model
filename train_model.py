from utils.config import *
from utils.run import run_experiment

if __name__ == '__main__':
    
    model, metrics = run_experiment(
        SEED,
        OUT_DIR
    )