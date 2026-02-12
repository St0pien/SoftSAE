# Import the W&B Python Library and log into W&B
import wandb

# 1: Define objective/training function
def objective(config):
    score = config.x**3 + config.y
    return score

def main():
    with wandb.init(project="SoftSAE") as run:
        score = objective(run.config)
        run.log({"score": score})

# 2: Define the search space
sweep_configuration = {
    "method": "bayes",
    "metric": {"goal": "minimize", "name": "score"},
    "parameters": {
        "x": {"max": 0.1, "min": 0.01},
        "y": {"values": [1, 3, 7]},
    },
}

# 3: Start the sweep
sweep_id = wandb.sweep(sweep=sweep_configuration, project="SoftSAE")

wandb.agent(sweep_id, function=main, count=10)