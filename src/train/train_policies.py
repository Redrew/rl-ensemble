import gym
import ray
from ray import tune
from ray.tune.logger import pretty_print
from ray.rllib.agents.ppo import PPOTrainer
from .meta_env import MetaEnv

def main(config, num_samples=1):
    # stop_criteria = lambda trial_id, result: result["episode_reward_mean"] > 100
    stop_criteria = {"training_iteration": 50}
    analysis = tune.run(config.get("algo", PPOTrainer), 
                        config=config, 
                        stop=stop_criteria, 
                        checkpoint_at_end=True, 
                        num_samples=num_samples)

    checkpoint_paths = []
    policies = []
    for trial in analysis.trials:
        checkpoints = analysis.get_trial_checkpoints_paths(trial)
        checkpoint_path = checkpoints[0][0]
        checkpoint_paths.append(checkpoint_path)

        Trainable = trial.get_trainable_cls()
        trainer = Trainable(trial.config)
        trainer.restore(checkpoint_path)
        policies.append(trainer.get_policy())
    
    return policies

if __name__ == "__main__":
    pass
    # ray.init()
    # main({'env': "CartPole-v0", "framework": "torch", "num_envs_per_worker": 10}, 5)