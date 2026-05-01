import os
import random
import numpy as np
import torch
import wandb
import tqdm
from absl import app, flags

from make_dmc_torch import make_env_dmc
from replay_buffer_torch import ReplayBuffer
from bro_torch import BRO

# ---------------------------------------------------------------------------
# Flags (matching JAX version where possible)
# ---------------------------------------------------------------------------
FLAGS = flags.FLAGS

flags.DEFINE_string('env_name', 'cheetah-run', 'Environment name.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_integer('eval_episodes', 5, 'Number of episodes used for evaluation.')
flags.DEFINE_integer('eval_interval', 25000, 'Eval interval.')
flags.DEFINE_integer('batch_size', 128, 'Mini batch size.')
flags.DEFINE_integer('max_steps', 1000000, 'Number of training steps.')
flags.DEFINE_integer('replay_buffer_size', 1000000, 'Replay buffer capacity.')
flags.DEFINE_integer('start_training', 2500, 'Number of training steps to start training.')
flags.DEFINE_integer('updates_per_step', 10, 'Number of updates per step (replay ratio).')
flags.DEFINE_boolean('distributional', True, 'Use distributional critic.')
flags.DEFINE_boolean('tqdm', False, 'Use tqdm progress bar.')
flags.DEFINE_string('save_dir', './results_torch/', 'Directory to save checkpoints and logs.')

def get_done(termination, truncation):
    return 1.0 if (termination or truncation) else 0.0

def evaluate(eval_env, agent, eval_episodes: int, seed: int):
    returns = np.zeros(eval_episodes)
    for episode in range(eval_episodes):
        observation, _ = eval_env.reset(seed=seed + episode + 1000) # Offset seed for eval
        episode_done = False
        while not episode_done:
            # Use conservative actor deterministically for evaluation
            action = agent.get_action(
                torch.from_numpy(observation).unsqueeze(0).to(agent.device),
                temperature=0.0
            )
            action = action.cpu().numpy()[0]
            next_observation, reward, termination, truncation, _ = eval_env.step(action)
            returns[episode] += reward
            observation = next_observation
            if termination or truncation:
                episode_done = True
    return {'return': returns.mean()}

def log_to_wandb_if_time_to(step, infos, eval_interval, suffix=''):
    if step % eval_interval == 0:
        dict_to_log = {'timestep': step}
        for info_key, value in infos.items():
            dict_to_log[f'{info_key}{suffix}'] = value
        wandb.log(dict_to_log, step=step)

def main(_):
    # -----------------------------------------------------------------------
    # Setup
    # -----------------------------------------------------------------------
    os.makedirs(FLAGS.save_dir, exist_ok=True)
    
    wandb.init(
        config=FLAGS,
        project='BRO_Torch',
        group=f'{FLAGS.env_name}',
        name=f'BRO_seed:{FLAGS.seed}_RR:{FLAGS.updates_per_step}'
    )
    
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    torch.manual_seed(FLAGS.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # -----------------------------------------------------------------------
    # Environments
    # -----------------------------------------------------------------------
    env = make_env_dmc(FLAGS.env_name)
    eval_env = make_env_dmc(FLAGS.env_name)
    
    obs_dim = env.observation_space.shape[-1]
    act_dim = env.action_space.shape[-1]

    # -----------------------------------------------------------------------
    # Agent & Buffer
    # -----------------------------------------------------------------------
    buffer = ReplayBuffer(
        buffer_size=FLAGS.replay_buffer_size, 
        observation_size=obs_dim, 
        action_size=act_dim, 
        device=device
    )
    
    agent = BRO(
        state_size=obs_dim, 
        action_size=act_dim, 
        device=device, 
        updates_per_step=FLAGS.updates_per_step,
        distributional=FLAGS.distributional
    )

    # -----------------------------------------------------------------------
    # Training Loop
    # -----------------------------------------------------------------------
    observation, _ = env.reset(seed=FLAGS.seed)
    
    for i in tqdm.tqdm(range(1, FLAGS.max_steps + 1), disable=not FLAGS.tqdm, smoothing=0.1):
        # 1. Select action
        if i <= FLAGS.start_training:
            action = env.action_space.sample()
        else:
            # Use optimistic actor stochastically for training
            action = agent.get_action_optimistic(
                torch.from_numpy(observation).unsqueeze(0).to(device),
                temperature=1.0
            )
            action = action.cpu().numpy()[0]
            
        # 2. Step environment
        next_observation, reward, termination, truncation, _ = env.step(action)
        done = get_done(termination, truncation)
        
        # 3. Store in buffer
        buffer.add(observation, next_observation, action, reward, done)
        observation = next_observation
        
        if termination or truncation:
            # Note: DMC tasks typically don't terminate, but just in case or if changed
            observation, _ = env.reset(seed=FLAGS.seed + i) # Vary seed slightly
            
        # 4. Update Agent
        if i > FLAGS.start_training:
            observations, next_observations, actions, rewards, dones = buffer.sample_multibatch(
                FLAGS.batch_size, FLAGS.updates_per_step
            )
            info = agent.update(
                i, observations, next_observations, actions, rewards, dones, FLAGS.updates_per_step
            )
            # Log train info
            log_to_wandb_if_time_to(i, info, FLAGS.eval_interval)
            
        # 5. Evaluate and Save
        if i % FLAGS.eval_interval == 0:
            eval_info = evaluate(eval_env, agent, FLAGS.eval_episodes, FLAGS.seed)
            log_to_wandb_if_time_to(i, eval_info, FLAGS.eval_interval, suffix='_eval')
            
            # Save checkpoint
            ckpt_path = os.path.join(FLAGS.save_dir, f'bro_{FLAGS.env_name}_step_{i}.pt')
            agent.save_checkpoint(ckpt_path)
            
    # Final checkpoint
    agent.save_checkpoint(os.path.join(FLAGS.save_dir, f'bro_{FLAGS.env_name}_final.pt'))

if __name__ == '__main__':
    app.run(main)
