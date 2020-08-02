import ray
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
from train import train_policies
from train.meta_env import MetaEnv

if __name__ == "__main__":
    ray.init()
    env_id = "HalfCheetah-v2"
    num_samples = 1

    config = {'env': env_id, "framework": "torch", 
              "num_envs_per_worker": 10, "num_gpus": 0.1}
    policies = train_policies.main(config, num_samples)

    tune.run("PG", config={"env": env_id, "framework": "torch",
                           "num_gpus_per_worker": 0.1, "num_gpus": 0.5, 
                           "env_config": {"base_env": env_id, "policies": policies},
                           "stop": {"training_iteration": 100}})

    from IPython import embed; embed(); exit(1)

    obs = env.reset()
    for i in range(1000):
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        env.base_env.render()
        if done: break
    env.close()
    
    model = A2C(MlpPolicy, "CartPole-v1", verbose=1, tensorboard_log="../tensorboard")
    model.learn(total_timesteps=10000)

    dummy_trainers = []
    for _ in range(5):
        dummy_trainers.append(PPOTrainer({"env": "CartPole-v1", "framework": "torch"}))


    # tune.run("PG", 
    #          name="EnsembleController",
    #          config={
    #              "env": "metaenv",
    #              "num_workers": 0,
    #              "num_gpus": 1},
    #          stop={"training_iteration": 20})

