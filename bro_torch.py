"""
Faithful PyTorch port of BRO (Bigger, Regularized, Optimistic).

Ported from the JAX version in jaxrl/bro/ and jaxrl/networks/.
Includes the full dual-actor design (conservative + optimistic),
KL-regularized exploration, and distributional quantile critic.
"""

import torch
import torch.nn as nn
import numpy as np
import copy


# ---------------------------------------------------------------------------
# Initialization helpers (matching JAX default_init with orthogonal)
# ---------------------------------------------------------------------------

def layer_init(layer, gain=None, bias_const=0.0):
    """Orthogonal init matching jaxrl default_init(scale)."""
    if gain is None:
        gain = np.sqrt(2)
    nn.init.orthogonal_(layer.weight, gain=gain)
    if layer.bias is not None:
        nn.init.constant_(layer.bias, bias_const)
    return layer


# ---------------------------------------------------------------------------
# BroNet backbone  (from jaxrl/networks/common.py  BroNet)
# ---------------------------------------------------------------------------

class BroNetBlock(nn.Module):
    """Residual block: Dense → LN → ReLU → Dense → LN  +  skip."""
    def __init__(self, hidden_size):
        super().__init__()
        self.block = nn.Sequential(
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.LayerNorm(hidden_size),
        )

    def forward(self, x):
        return x + self.block(x)


class BroNet(nn.Module):
    """Projection + N residual blocks + optional output head."""
    def __init__(self, input_size, output_size, hidden_size, num_blocks):
        super().__init__()
        self.projection = nn.Sequential(
            layer_init(nn.Linear(input_size, hidden_size)),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
        )
        self.blocks = nn.ModuleList([BroNetBlock(hidden_size) for _ in range(num_blocks)])
        self.output_size = output_size
        if output_size is not None:
            self.final_layer = layer_init(nn.Linear(hidden_size, output_size))

    def forward(self, x):
        x = self.projection(x)
        for block in self.blocks:
            x = block(x)
        if self.output_size is not None:
            x = self.final_layer(x)
        return x


# ---------------------------------------------------------------------------
# Critic  (from jaxrl/networks/critic_net.py)
# ---------------------------------------------------------------------------

class BroNetCritic(nn.Module):
    def __init__(self, state_size, action_size, output_size, hidden_size=512, num_blocks=2):
        super().__init__()
        self.net = BroNet(state_size + action_size, output_size, hidden_size, num_blocks)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.net(x)


class BroNetCritics(nn.Module):
    """Double critic (two independent BroNet critics)."""
    def __init__(self, state_size, action_size, output_size, hidden_size=512, num_blocks=2):
        super().__init__()
        self.critic1 = BroNetCritic(state_size, action_size, output_size, hidden_size, num_blocks)
        self.critic2 = BroNetCritic(state_size, action_size, output_size, hidden_size, num_blocks)

    def forward(self, state, action):
        return self.critic1(state, action), self.critic2(state, action)


# ---------------------------------------------------------------------------
# Conservative Actor  (from jaxrl/networks/policies.py  NormalTanhPolicy)
# ---------------------------------------------------------------------------

class BroNetActor(nn.Module):
    """Conservative actor: obs → (mean, std) for a TanhNormal policy."""
    def __init__(self, state_size, action_size, hidden_size=256, num_blocks=1,
                 log_std_min=-10.0, log_std_max=2.0, log_std_scale=1.0):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.embedding = BroNet(state_size, output_size=None, hidden_size=hidden_size, num_blocks=num_blocks)
        self.means = layer_init(nn.Linear(hidden_size, action_size))
        self.log_stds = layer_init(nn.Linear(hidden_size, action_size), gain=log_std_scale)

    def forward(self, state, temperature=1.0):
        x = self.embedding(state)
        means = self.means(x)
        log_stds = self.log_stds(x)
        # Soft clamping via tanh (matching JAX version)
        log_stds = self.log_std_min + (self.log_std_max - self.log_std_min) * 0.5 * (1 + torch.tanh(log_stds))
        stds = torch.exp(log_stds) * (temperature + 1e-8)
        return means, stds


# ---------------------------------------------------------------------------
# Optimistic Actor  (from jaxrl/networks/policies.py  DualTanhPolicy)
# ---------------------------------------------------------------------------

class BroNetOptimisticActor(nn.Module):
    """Optimistic actor: shifts the conservative actor's mean, scales its std."""
    def __init__(self, state_size, action_size, hidden_size=256, num_blocks=1, scale_means=0.01):
        super().__init__()
        # Input: concat(obs, conservative_means)
        self.embedding = BroNet(state_size + action_size, output_size=None, hidden_size=hidden_size, num_blocks=num_blocks)
        self.action_shift = nn.Linear(hidden_size, action_size, bias=False)
        nn.init.orthogonal_(self.action_shift.weight, gain=scale_means)

    def forward(self, observations, means_c, stds_c, std_multiplier):
        inputs = torch.cat([observations, means_c], dim=-1)
        x = self.embedding(inputs)
        shift = self.action_shift(x)
        optimistic_means = means_c + shift
        optimistic_stds = stds_c * std_multiplier
        return optimistic_means, optimistic_stds


# ---------------------------------------------------------------------------
# Sampling helpers (TanhNormal)
# ---------------------------------------------------------------------------

def sample_tanh_normal(means, stds, return_log_prob=True):
    """Sample from TanhNormal, return (action, log_prob)."""
    normal = torch.distributions.Normal(means, stds)
    x_t = normal.rsample()
    action = torch.tanh(x_t)
    if return_log_prob:
        log_prob = normal.log_prob(x_t)
        # Tanh correction
        log_prob -= torch.log(1.0 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1)  # sum over action dims → (batch,)
        return action, log_prob
    return action, None


# ---------------------------------------------------------------------------
# Quantile Huber Loss  (from jaxrl/bro/critic.py)
# ---------------------------------------------------------------------------

def huber_replace(td_errors, kappa=1.0):
    return torch.where(
        torch.abs(td_errors) <= kappa,
        0.5 * td_errors ** 2,
        kappa * (torch.abs(td_errors) - 0.5 * kappa),
    )


def calculate_quantile_huber_loss(td_errors, taus, kappa=1.0):
    element_wise_huber_loss = huber_replace(td_errors, kappa)
    mask = torch.where(td_errors < 0, 1.0, 0.0).detach()
    element_wise_quantile_huber_loss = torch.abs(taus[..., None] - mask) * element_wise_huber_loss / kappa
    return element_wise_quantile_huber_loss.sum(dim=1).mean()


# ---------------------------------------------------------------------------
# Adjustment module  (from jaxrl/bro/temperature.py)
# ---------------------------------------------------------------------------

class Adjustment(nn.Module):
    """Learnable Lagrange multiplier with tanh-bounded log-space parameterization."""
    def __init__(self, init_value=1.0, log_val_min=-10.0, log_val_max=7.5):
        super().__init__()
        self.log_val_min = log_val_min
        self.log_val_max = log_val_max
        # Compute raw param so that forward() initially returns init_value
        raw_init = self._inverse_transform(init_value)
        self.log_value = nn.Parameter(torch.tensor(raw_init, dtype=torch.float32))

    def _inverse_transform(self, target_value):
        """Find raw param value that maps to target_value through the transform."""
        log_target = np.log(target_value)
        # tanh(raw) = 2*(log_target - log_val_min)/(log_val_max - log_val_min) - 1
        tanh_val = 2.0 * (log_target - self.log_val_min) / (self.log_val_max - self.log_val_min) - 1.0
        tanh_val = np.clip(tanh_val, -0.999, 0.999)
        return np.arctanh(tanh_val)

    def forward(self):
        log_val = self.log_val_min + (self.log_val_max - self.log_val_min) * 0.5 * (1 + torch.tanh(self.log_value))
        return torch.exp(log_val)


# ---------------------------------------------------------------------------
# BRO Agent  (from jaxrl/bro/bro_learner.py)
# ---------------------------------------------------------------------------

class BRO(nn.Module):
    def __init__(
        self,
        state_size: int,
        action_size: int,
        device: str,
        # Learning rates
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        temp_lr: float = 3e-4,
        adj_lr: float = 3e-5,
        # SAC params
        discount: float = 0.99,
        tau: float = 0.005,
        target_entropy: float = None,
        init_temperature: float = 1.0,
        pessimism: float = 0.0,
        # BRO-specific
        updates_per_step: int = 2,
        distributional: bool = True,
        n_quantiles: int = 100,
        kl_target: float = 0.05,
        std_multiplier: float = 0.75,
        init_optimism: float = 1.0,
        init_regularizer: float = 0.25,
        use_compile: bool = False,
    ):
        super().__init__()
        self.device = device
        self.state_size = state_size
        self.action_size = action_size
        self.discount = discount
        self.tau = tau
        self.pessimism = pessimism
        self.distributional = distributional
        self.n_quantiles = n_quantiles if distributional else 1
        self.kl_target = kl_target
        self.std_multiplier = std_multiplier
        self.target_entropy = -action_size / 2.0 if target_entropy is None else target_entropy
        self.init_temperature = init_temperature
        self.init_optimism = init_optimism
        self.init_regularizer = init_regularizer
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.temp_lr = temp_lr
        self.adj_lr = adj_lr
        self.use_compile = use_compile

        # Quantile taus
        quantile_taus = torch.arange(0, n_quantiles + 1, dtype=torch.float32) / n_quantiles
        self.quantile_taus = ((quantile_taus[1:] + quantile_taus[:-1]) / 2.0).unsqueeze(0).to(device)

        # Reset schedule (matching JAX version)
        self.reset_list = [15001, 50001, 250001, 500001, 750001, 1000001, 1500001, 2000001]
        if updates_per_step == 2:
            self.reset_list = self.reset_list[:1]

        self.reset()

    def reset(self):
        """Full re-initialization of all networks and optimizers."""
        d = self.device

        # --- Networks ---
        self.critic = BroNetCritics(self.state_size, self.action_size, self.n_quantiles, 512, 2).to(d)
        self.target_critic = BroNetCritics(self.state_size, self.action_size, self.n_quantiles, 512, 2).to(d)
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor = BroNetActor(self.state_size, self.action_size, 256, 1).to(d)
        self.actor_o = BroNetOptimisticActor(self.state_size, self.action_size, 256, 1).to(d)

        if self.use_compile:
            print("Optimizing networks with torch.compile...")
            self.critic = torch.compile(self.critic)
            self.target_critic = torch.compile(self.target_critic)
            self.actor = torch.compile(self.actor)
            self.actor_o = torch.compile(self.actor_o)

        # Temperature (learnable log param)
        self.log_temp = nn.Parameter(torch.tensor(np.log(self.init_temperature), dtype=torch.float32, device=d))

        # Adjustment modules (optimism & regularizer)
        self.optimism = Adjustment(self.init_optimism).to(d)
        self.regularizer = Adjustment(self.init_regularizer).to(d)

        # --- Optimizers (matching JAX: adamw for actor/critic, adam(b1=0.5) for temp/adj) ---
        self.optimizer_critic = torch.optim.AdamW(self.critic.parameters(), lr=self.critic_lr, weight_decay=1e-4)
        self.optimizer_actor = torch.optim.AdamW(self.actor.parameters(), lr=self.actor_lr, weight_decay=1e-4)
        self.optimizer_actor_o = torch.optim.AdamW(self.actor_o.parameters(), lr=self.actor_lr, weight_decay=1e-4)
        self.optimizer_log_temp = torch.optim.Adam([self.log_temp], lr=self.temp_lr, betas=(0.5, 0.999))
        self.optimizer_optimism = torch.optim.Adam(self.optimism.parameters(), lr=self.adj_lr, betas=(0.5, 0.999))
        self.optimizer_regularizer = torch.optim.Adam(self.regularizer.parameters(), lr=self.adj_lr, betas=(0.5, 0.999))

    @property
    def temp(self):
        return self.log_temp.exp()

    # ------------------------------------------------------------------
    # Action sampling
    # ------------------------------------------------------------------

    @torch.no_grad()
    def get_action(self, state, temperature=1.0):
        """Sample from conservative actor (used during evaluation)."""
        means, stds = self.actor(state, temperature=temperature)
        if temperature < 1e-8:
            return torch.tanh(means)
        action, _ = sample_tanh_normal(means, stds, return_log_prob=False)
        return action

    @torch.no_grad()
    def get_action_optimistic(self, state, temperature=1.0):
        """Sample from optimistic actor (used during training)."""
        means_c, stds_c = self.actor(state, temperature=temperature)
        means_o, stds_o = self.actor_o(state, means_c, stds_c, self.std_multiplier)
        if temperature < 1e-8:
            return torch.tanh(means_o)
        action, _ = sample_tanh_normal(means_o, stds_o, return_log_prob=False)
        return action

    # ------------------------------------------------------------------
    # Critic update  (from jaxrl/bro/critic.py)
    # ------------------------------------------------------------------

    def update_critic_distributional(self, observations, next_observations, actions, rewards, dones):
        kappa = 1.0
        with torch.no_grad():
            means, stds = self.actor(next_observations)
            next_actions, next_log_probs = sample_tanh_normal(means, stds)
            next_q1, next_q2 = self.target_critic(next_observations, next_actions)
            next_q = (next_q1 + next_q2) / 2 - self.pessimism * torch.abs(next_q1 - next_q2) / 2
            # target_q shape: (batch, n_quantiles_target, n_quantiles)
            target_q = rewards[:, None, None] + self.discount * (1 - dones[:, None, None]) * next_q[:, None, :]
            target_q -= self.discount * self.temp * (1 - dones[:, None, None]) * next_log_probs[:, None, None]

        q1, q2 = self.critic(observations, actions)
        td_errors1 = target_q - q1[:, :, None]
        td_errors2 = target_q - q2[:, :, None]
        critic_loss = calculate_quantile_huber_loss(td_errors1, self.quantile_taus, kappa) + \
                      calculate_quantile_huber_loss(td_errors2, self.quantile_taus, kappa)

        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        self.optimizer_critic.step()
        return {
            'critic_loss': critic_loss.item(),
            'q1': q1.mean().item(),
            'q2': q2.mean().item(),
        }

    def update_critic_standard(self, observations, next_observations, actions, rewards, dones):
        with torch.no_grad():
            means, stds = self.actor(next_observations)
            next_actions, next_log_probs = sample_tanh_normal(means, stds)
            next_q1, next_q2 = self.target_critic(next_observations, next_actions)
            next_q = (next_q1 + next_q2) / 2 - self.pessimism * torch.abs(next_q1 - next_q2) / 2
            target_q = rewards.unsqueeze(-1) + self.discount * (1 - dones.unsqueeze(-1)) * next_q
            target_q -= self.discount * self.temp * (1 - dones.unsqueeze(-1)) * next_log_probs.unsqueeze(-1)

        q1, q2 = self.critic(observations, actions)
        critic_loss = ((q1 - target_q) ** 2 + (q2 - target_q) ** 2).mean()

        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        self.optimizer_critic.step()
        return {
            'critic_loss': critic_loss.item(),
            'q1': q1.mean().item(),
            'q2': q2.mean().item(),
        }

    # ------------------------------------------------------------------
    # Conservative actor update  (from jaxrl/bro/actor.py  update)
    # ------------------------------------------------------------------

    def update_actor(self, observations):
        means, stds = self.actor(observations)
        actions, log_probs = sample_tanh_normal(means, stds)
        q1, q2 = self.critic(observations, actions)
        q = (q1 + q2) / 2 - self.pessimism * torch.abs(q1 - q2) / 2
        if self.distributional:
            q = q.mean(dim=-1)
        actor_loss = (self.temp.detach() * log_probs - q).mean()

        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        self.optimizer_actor.step()

        entropy = -log_probs.mean().item()
        return {
            'actor_loss': actor_loss.item(),
            'entropy': entropy,
            'std': stds.mean().item(),
        }, -log_probs.mean().detach()  # return entropy tensor for temp update

    # ------------------------------------------------------------------
    # Optimistic actor update  (from jaxrl/bro/actor.py  update_optimistic)
    # ------------------------------------------------------------------

    def update_actor_optimistic(self, observations):
        with torch.no_grad():
            means_c, stds_c = self.actor(observations, temperature=1.0)

        means_o, stds_o = self.actor_o(observations, means_c, stds_c, self.std_multiplier)
        actions, _ = sample_tanh_normal(means_o, stds_o)
        q1, q2 = self.critic(observations, actions)
        q_ub = (q1 + q2) / 2 + self.optimism().detach() * torch.abs(q1 - q2) / 2
        if self.distributional:
            q_ub = q_ub.mean(dim=-1)

        # Analytical KL(optimistic_base || conservative)
        std_ratio = stds_o / self.std_multiplier  # undo the std scaling
        kl = (torch.log(stds_c / std_ratio)
              + (std_ratio ** 2 + (means_o - means_c) ** 2) / (2 * stds_c ** 2)
              - 0.5).sum(dim=-1)

        actor_o_loss = (-q_ub).mean() + self.regularizer().detach() * kl.mean()

        self.optimizer_actor_o.zero_grad()
        actor_o_loss.backward()
        self.optimizer_actor_o.step()

        return {
            'actor_o_loss': actor_o_loss.item(),
            'kl': kl.mean().item(),
            'std_o': stds_o.mean().item(),
            'Q_mean': ((q1 + q2) / 2).mean().item(),
            'Q_std': (torch.abs(q1 - q2) / 2).mean().item(),
        }, kl.mean().detach() / self.action_size  # empirical_kl for Lagrange updates

    # ------------------------------------------------------------------
    # Temperature & Lagrange multiplier updates
    # ------------------------------------------------------------------

    def update_temperature(self, entropy):
        temp_loss = self.temp * (entropy - self.target_entropy).detach()
        self.optimizer_log_temp.zero_grad()
        temp_loss.backward()
        self.optimizer_log_temp.step()
        return {'temperature': self.temp.item(), 'temp_loss': temp_loss.item()}

    def update_optimism(self, empirical_kl):
        optimism_val = self.optimism()
        optimism_loss = (optimism_val - self.pessimism) * (empirical_kl - self.kl_target)
        self.optimizer_optimism.zero_grad()
        optimism_loss.backward()
        self.optimizer_optimism.step()
        return {'optimism': optimism_val.item(), 'optimism_loss': optimism_loss.item()}

    def update_regularizer(self, empirical_kl):
        kl_weight = self.regularizer()
        regularizer_loss = -kl_weight * (empirical_kl - self.kl_target)
        self.optimizer_regularizer.zero_grad()
        regularizer_loss.backward()
        self.optimizer_regularizer.step()
        return {'kl_weight': kl_weight.item(), 'regularizer_loss': regularizer_loss.item()}

    # ------------------------------------------------------------------
    # Target critic soft update
    # ------------------------------------------------------------------

    def update_target_critic(self):
        for p, tp in zip(self.critic.parameters(), self.target_critic.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

    # ------------------------------------------------------------------
    # Single update step  (one gradient step for all components)
    # ------------------------------------------------------------------

    def single_update(self, observations, next_observations, actions, rewards, dones):
        # 1. Critic
        if self.distributional:
            critic_info = self.update_critic_distributional(observations, next_observations, actions, rewards, dones)
        else:
            critic_info = self.update_critic_standard(observations, next_observations, actions, rewards, dones)
        self.update_target_critic()

        # 2. Conservative actor
        actor_info, entropy = self.update_actor(observations)

        # 3. Optimistic actor
        actor_o_info, empirical_kl = self.update_actor_optimistic(observations)

        # 4. Temperature
        temp_info = self.update_temperature(entropy)

        # 5. Lagrange multipliers
        optimism_info = self.update_optimism(empirical_kl)
        regularizer_info = self.update_regularizer(empirical_kl)

        return {**critic_info, **actor_info, **actor_o_info, **temp_info, **optimism_info, **regularizer_info}

    # ------------------------------------------------------------------
    # Full update  (multiple steps per env step + periodic reset)
    # ------------------------------------------------------------------

    def update(self, step, observations, next_observations, actions, rewards, dones, num_updates):
        if step in self.reset_list:
            self.reset()
        info = {}
        for i in range(num_updates):
            info = self.single_update(
                observations[i], next_observations[i], actions[i], rewards[i], dones[i]
            )
        return info

    # ------------------------------------------------------------------
    # Checkpoint save/load
    # ------------------------------------------------------------------

    def save_checkpoint(self, path):
        torch.save({
            'critic': self.critic.state_dict(),
            'target_critic': self.target_critic.state_dict(),
            'actor': self.actor.state_dict(),
            'actor_o': self.actor_o.state_dict(),
            'log_temp': self.log_temp.data,
            'optimism': self.optimism.state_dict(),
            'regularizer': self.regularizer.state_dict(),
        }, path)

    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.critic.load_state_dict(ckpt['critic'])
        self.target_critic.load_state_dict(ckpt['target_critic'])
        self.actor.load_state_dict(ckpt['actor'])
        self.actor_o.load_state_dict(ckpt['actor_o'])
        self.log_temp.data = ckpt['log_temp']
        self.optimism.load_state_dict(ckpt['optimism'])
        self.regularizer.load_state_dict(ckpt['regularizer'])
