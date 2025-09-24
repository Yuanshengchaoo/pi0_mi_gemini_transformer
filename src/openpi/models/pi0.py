# src/openpi/models/pi0.py

import dataclasses
import logging
import math

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

logger = logging.getLogger("openpi")


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision."""
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


# --- Start: New code for MINE ---

def ema(mu, alpha, past_ema):
    """Exponential moving average."""
    return alpha * mu + (1.0 - alpha) * past_ema

# def ema_loss(x: at.Array, running_mean_var: nnx.Variable, alpha: float) -> at.Array:
#     """Computes the exponential moving average of the log-sum-exp of scores."""
#     t_exp = jnp.exp(logsumexp(x, axis=0) - math.log(x.shape[0]))

#     is_initialized = running_mean_var.value != 0.0
#     new_ema = jax.lax.cond(
#         is_initialized,
#         lambda: ema(t_exp, alpha, running_mean_var.value),
#         lambda: t_exp,
#     )
    
#     running_mean_var.value = new_ema
    
#     t_log = jnp.log(running_mean_var.value + 1e-6)
#     return t_log

# --- 关键修改：在 ema_loss 中使用 stop_gradient ---
def ema_loss(x: at.Array, running_mean_var: nnx.Variable, alpha: float) -> at.Array:
    """Computes the exponential moving average of the log-sum-exp of scores."""
    t_exp = jnp.exp(logsumexp(x, axis=0) - math.log(x.shape[0]))

    is_initialized = running_mean_var.value != 0.0
    new_ema = jax.lax.cond(
        is_initialized,
        lambda: ema(t_exp, alpha, running_mean_var.value),
        lambda: t_exp,
    )
    
    # Update the state with the new EMA value
    running_mean_var.value = new_ema
    
    # Use stop_gradient to prevent gradients from flowing through the EMA history
    t_log = jnp.log(jax.lax.stop_gradient(new_ema) + 1e-6)
    return t_log


class CrossAttentionCritic(nnx.Module):
    """A Transformer-based critic using cross-attention."""
    def __init__(self, embed_dim: int, num_heads: int, *, rngs: nnx.Rngs):
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.cross_attention = nnx.MultiHeadAttention(
            num_heads=self.num_heads,
            in_features=self.embed_dim,
            rngs=rngs,
        )
        # --- MODIFICATION START ---
        # Replaced nnx.Sequential with named layers to avoid integer keys in param paths.
        self.mlp_linear1 = nnx.Linear(self.embed_dim, self.embed_dim, rngs=rngs)
        self.mlp_linear2 = nnx.Linear(self.embed_dim, 1, rngs=rngs)
        # --- MODIFICATION END ---
        self.ln_x = nnx.LayerNorm(self.embed_dim, rngs=rngs)
        self.ln_z = nnx.LayerNorm(self.embed_dim, rngs=rngs)

    def __call__(self, x, z):
        """
        Args:
            x: Tokens for Key/Value, shape (B, L_x, D).
            z: Tokens for Query, shape (B, L_z, D).
        Returns:
            A scalar score `t` for each item in the batch, shape (B,).
        """
        norm_x = self.ln_x(x)
        norm_z = self.ln_z(z)
        
        attn_output = self.cross_attention(inputs_q=norm_z, inputs_k=norm_x, inputs_v=norm_x,decode=False)
        
        # --- MODIFICATION START ---
        # Manually apply the layers in sequence.
        mlp_out = self.mlp_linear1(attn_output)
        mlp_out = nnx.gelu(mlp_out)
        t_tokens = self.mlp_linear2(mlp_out)
        # --- MODIFICATION END ---
        
        t = jnp.mean(t_tokens, axis=1)
        return t.squeeze(axis=-1)

class Mine(nnx.Module):
    """JAX/NNX implementation of the MINE estimator for Mutual Information."""
    def __init__(self, critic: nnx.Module, loss_type: str = "mine", alpha: float = 0.01):
        self.critic = critic
        self.running_mean = nnx.Variable(0.0)
        self.loss_type = loss_type
        self.alpha = alpha

    def __call__(self, x: at.Array, z: at.Array, *, key: at.Key, z_marg: at.Array | None = None) -> at.Array:
        if z_marg is None:
            perm = jax.random.permutation(key, z.shape[0])
            z_marg = z[perm]
        
        t = self.critic(x, z)
        t_marg = self.critic(x, z_marg)

        if self.loss_type == "mine":
            second_term = ema_loss(t_marg, self.running_mean, self.alpha)
            mi_est = jnp.mean(t) - second_term
        elif self.loss_type == "fdiv":
            second_term = jnp.mean(jnp.exp(t_marg - 1))
            mi_est = jnp.mean(t) - second_term
        elif self.loss_type == "mine_biased":
            second_term = logsumexp(t_marg, axis=0) - math.log(t_marg.shape[0])
            mi_est = jnp.mean(t) - second_term
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")
        
        return -mi_est

    def get_mi(self, x: at.Array, z: at.Array, *, key: at.Key, z_marg: at.Array | None = None) -> at.Array:
        return -self(x, z, key=key, z_marg=z_marg)

# --- End: New code for MINE ---


@dataclasses.dataclass(frozen=True)
class Pi0Config(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b_lora"
    action_expert_variant: _gemma.Variant = "gemma_300m_lora"

    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = 48
    
    mi_beta: float = 1.0
    critic_num_heads: int = 8

    @property
    @override
    def model_type(self) -> _model.ModelType:
        return _model.ModelType.PI0

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0":
        return Pi0(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        filters = []
        has_lora = False
        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")
        if "lora" in self.paligemma_variant:
            filters.append(
                gemma_params_filter,
            )
            if "lora" not in self.action_expert_variant:
                filters.append(
                    nnx.Not(action_expert_params_filter),
                )
            has_lora = True
        elif "lora" in self.action_expert_variant:
            filters.append(
                action_expert_params_filter,
            )
            has_lora = True

        if has_lora:
            filters.append(
                nnx.Not(nnx_utils.PathRegex(".*lora.*")),
            )
        if not filters:
            return nnx.Nothing
        return nnx.All(*filters)


class Pi0(_model.BaseModel):
    def __init__(self, config: Pi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.config = config
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
            )
        )
        llm.lazy_init(rngs=rngs, method="init")
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)
        
        if self.config.mi_beta > 0:
            critic = CrossAttentionCritic(
                embed_dim=paligemma_config.width,
                num_heads=self.config.critic_num_heads,
                rngs=rngs,
            )
            self.mine = Mine(critic=critic)

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)

            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            ar_mask += [False] * image_tokens.shape[1]

        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            ar_mask += [False] * tokenized_inputs.shape[1]
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    @at.typecheck
    def embed_suffix(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []
        state_token = self.state_proj(obs.state)[:, None, :]
        tokens.append(state_token)
        input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
        ar_mask += [True]

        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        action_tokens = self.action_in_proj(noisy_actions)
        time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
        action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
        action_time_tokens = self.action_time_mlp_in(action_time_tokens)
        action_time_tokens = nnx.swish(action_time_tokens)
        action_time_tokens = self.action_time_mlp_out(action_time_tokens)
        tokens.append(action_time_tokens)
        input_mask.append(jnp.ones(action_time_tokens.shape[:2], dtype=jnp.bool_))
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        preprocess_rng, noise_rng, time_rng, mine_rng = jax.random.split(rng, 4)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask = self.embed_suffix(observation, x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

        l2_loss = jnp.mean(jnp.square(v_t - u_t), axis=-1)

        if self.config.mi_beta > 0:
            mi_loss = self.mine(prefix_tokens, prefix_out, key=mine_rng)
            jax.debug.print("mi_est:{}", -1*mi_loss)
            total_loss = l2_loss - self.config.mi_beta * mi_loss
            # jax.debug.print("mse_loss:{}", l2_loss)
            return total_loss
        else:
            return l2_loss

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
    ) -> _model.Actions:
        observation = _model.preprocess_observation(None, observation, train=False)
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens], mask=full_attn_mask, positions=positions, kv_cache=kv_cache
            )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0