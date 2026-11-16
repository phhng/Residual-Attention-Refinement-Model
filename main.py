from utils.config import *
from utils.run import *

if __name__ == '__main__':
    
    model, metrics = run_experiment(
        SEED,
        OUT_DIR
    )

    # res, summary = run_multi_seed()