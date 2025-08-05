from typing import Any, Dict, Tuple, Union

import torch
from deepspeed import DeepSpeedEngine
from transformers import PreTrainedModel
import torch.nn.functional as F

from utils import compute_token_log_probs


def compute_pg_loss(
    policy_model: Union[DeepSpeedEngine, PreTrainedModel],
    reference_model: Union[DeepSpeedEngine, PreTrainedModel],
    batch: Dict[str, torch.Tensor],
    total_response_len: int,
    TEMPERATURE: float,
    KL_COEFFICIENT: float,
    algo_config: Dict[str, Any] = None, 
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute the policy gradient loss with KL penalty between policy and reference models.
    Supports GRPO, Dr. GRPO, and DAPO variants.

    This function:
    1. Computes log probabilities for both policy and reference models
    2. Calculates importance sampling ratio between current and old policy
    3. Implements clipping with configurable low/high bounds
    4. Optionally adds KL divergence penalty
    5. Supports various normalization schemes for advantages and length

    Args:
        policy_model: The model being trained
        reference_model: The reference model for KL penalty calculation
        batch: Dictionary containing:
            - input_ids: Tensor of shape [batch_size, seq_len]
            - attention_mask: Tensor of shape [batch_size, seq_len]
            - labels: Tensor of shape [batch_size, seq_len] with -100 for ignored positions
            - advantages: Tensor of shape [batch_size, seq_len]
            - old_logps: Optional tensor of shape [batch_size, seq_len-1] with old log probs
            - adv_den: Optional tensor with advantage denominators for Dr. GRPO/DAPO

        algo_config: Configuration for the algorithm variant:
            - eps_low: Lower clipping bound (default: 0.2)
            - eps_high: Higher clipping bound (default: eps_low or 0.28 for DAPO)
            - norm_adv: Whether to normalize advantages by std (default: "std" for GRPO, "none" for Dr. GRPO/DAPO)
            - length_norm: Whether to use response-level length normalization (default: True for GRPO, False for Dr. GRPO/DAPO)

    Returns:
        Tuple containing:
            - loss: Combined policy gradient and KL penalty loss (scalar tensor)
            - metrics: Dictionary with detailed loss components
    """
    # Set default configuration if not provided
    if algo_config is None:
        algo_config = {
            "eps_low": 0.2,
            "eps_high": 0.2,
            "norm_adv": "std",  # "std" or "none"
            "length_norm": True,  # True for GRPO, False for Dr. GRPO/DAPO
        }

    input_ids = batch["input_ids"]  # [batch_size, seq_len]
    attention_mask = batch["attention_mask"]  # [batch_size, seq_len]
    labels = batch["labels"]  # [batch_size, seq_len]
    advantages = batch["advantages"]  # [batch_size, seq_len]

    model_inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

    labels_mask = (labels[..., 1:] != -100).float()  # [batch_size, seq_len-1]

    # Compute reference log probabilities for KL penalty
    with torch.no_grad():
        ref_logps = compute_token_log_probs(
            reference_model, model_inputs, TEMPERATURE
        )  # [batch_size, seq_len-1]

    # Compute current log probabilities
    logps = compute_token_log_probs(
        policy_model, model_inputs, TEMPERATURE
    )  # [batch_size, seq_len-1]

    # Compute importance sampling ratio (if old_logps are available)
    if "old_logps" in batch:
        # Use stored old log probabilities (GRPO/Dr. GRPO/DAPO)
        old_logps = batch["old_logps"][..., 1:]
        ratio = torch.exp(logps - old_logps)
    else:
        # Fallback to policy gradient (no ratio/clipping)
        ratio = torch.ones_like(logps)

    # Advantage normalization is controlled by algo_config and already done in process_training_episodes
    adv = advantages[..., 1:]
    # No secondary normalization needed here

    # Compute clipped surrogate objective
    clipped_ratio = torch.clamp(
        ratio, min=1.0 - algo_config["eps_low"], max=1.0 + algo_config["eps_high"]
    )

    # Sign-aware clipping: use min for positive advantages, max for negative advantages
    use_min = adv >= 0
    surrogate1 = ratio * adv * labels_mask
    surrogate2 = clipped_ratio * adv * labels_mask
    policy_loss_per_token = -torch.where(
        use_min, torch.min(surrogate1, surrogate2), torch.max(surrogate1, surrogate2)
    )

    # Compute KL penalty separately (not inside surrogate clipping)
    kl_penalty = torch.exp(ref_logps - logps) - (ref_logps - logps) - 1
    kl_penalty = kl_penalty * labels_mask

    # Compute entropy for monitoring
    entropy = -logps.sum() / labels_mask.sum()

    # Length normalization is controlled by algo_config
    if algo_config["length_norm"]:
        # Original GRPO with response-level length normalization
        # Properly divide each response's loss by its length before averaging
        tok_per_resp = labels_mask.sum(-1)  # [B]
        policy_loss = (
            policy_loss_per_token.sum(-1) / tok_per_resp.clamp(min=1.0)
        ).mean()
    else:
        # Dr. GRPO / DAPO with token-level normalization
        if "adv_den" in batch:
            # Use provided token budget (Dr. GRPO)
            policy_loss = policy_loss_per_token.sum() / batch["adv_den"].sum()
        else:
            # Fallback to total response length (similar to Dr. GRPO)
            policy_loss = policy_loss_per_token.sum() / total_response_len

    # Apply KL penalty (separately from surrogate clipping)
    loss = policy_loss + KL_COEFFICIENT * kl_penalty.sum() / total_response_len

    # Compute metrics for clip rates - masked to only include valid response tokens
    with torch.no_grad():
        clip_low_rate = (
            (ratio < 1.0 - algo_config["eps_low"]) & (adv < 0) & (labels_mask > 0)
        ).float().sum() / labels_mask.sum()
        clip_high_rate = (
            (ratio > 1.0 + algo_config["eps_high"]) & (adv > 0) & (labels_mask > 0)
        ).float().sum() / labels_mask.sum()
        clip_rate = (
            ((ratio < 1.0 - algo_config["eps_low"]) & (adv < 0) & (labels_mask > 0))
            | ((ratio > 1.0 + algo_config["eps_high"]) & (adv > 0) & (labels_mask > 0))
        ).float().sum() / labels_mask.sum()

    metrics = {
        "policy_loss": policy_loss.item(),
        "kl_penalty": kl_penalty.sum().item() / total_response_len,
        "entropy": entropy.item(),
        "clip_ratio/low_rate": clip_low_rate.item(),
        "clip_ratio/high_rate": clip_high_rate.item(),
        "clip_ratio/region_rate": clip_rate.item(),
    }

    return loss, metrics

def compute_Q_mc_loss(
    policy_model: Union[DeepSpeedEngine, PreTrainedModel],
    reference_model: Union[DeepSpeedEngine, PreTrainedModel],
    batch: Dict[str, torch.Tensor],
    total_response_len: int,
    TEMPERATURE: float,
    algo_config: Dict[str, Any] = None, 
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute the Q-learning loss using Monte Carlo.
    """
    # get logits from policy model and reference model
    input_ids = batch["input_ids"]  # [batch_size, seq_len]
    attention_mask = batch["attention_mask"]  # [batch_size, seq_len]
    labels = batch["labels"]  # [batch_size, seq_len]
    advantages = batch["advantages"]  # [batch_size, seq_len]

    model_inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

    labels_mask = (labels[..., 1:] != -100).float()  # [batch_size, seq_len-1]

    # Compute reference log probabilities for KL penalty
    with torch.no_grad():
        ref_logps = compute_token_log_probs(
            reference_model, model_inputs, TEMPERATURE
        )  # [batch_size, seq_len-1]

    # Compute current log probabilities
    logps = compute_token_log_probs(
        policy_model, model_inputs, TEMPERATURE
    )  # [batch_size, seq_len-1]

    # Compute Q-function logits
    # Use learned parameters from model if available, otherwise use algo_config values
    Q_A = getattr(policy_model, 'Q_A', algo_config["Q_A"])
    Q_c = getattr(policy_model, 'Q_c', algo_config["Q_c"])
    
    # Convert to tensor and move to correct device if needed
    if not isinstance(Q_A, torch.Tensor):
        Q_A = torch.tensor(Q_A, dtype=torch.bfloat16, device=logps.device)
    if not isinstance(Q_c, torch.Tensor):
        Q_c = torch.tensor(Q_c, dtype=torch.bfloat16, device=logps.device)
    
    Q_logits = (1/algo_config["Q_beta"]) * (logps - ref_logps) + Q_A * ref_logps + Q_c

    # Handle split rewards if enabled
    if algo_config.get("split_rewards", False):
        # Extract format and correctness rewards from the batch
        reward_metrics = batch.get("reward_metrics", {})
        format_rewards = []
        correctness_rewards = []
        
        # Get the list of reward dictionaries for each sample
        if "format_reward" in reward_metrics:
            format_rewards = reward_metrics["format_reward"]
            # Handle different correctness reward names based on task
            if "correctness_reward" in reward_metrics:
                correctness_rewards = reward_metrics["correctness_reward"]
            elif "gsm8k_correctness_reward" in reward_metrics:
                correctness_rewards = reward_metrics["gsm8k_correctness_reward"]
            else:
                # Fallback: use total rewards minus format rewards
                total_rewards = batch["rewards"]
                correctness_rewards = [total - format for total, format in zip(total_rewards, format_rewards)]
        else:
            # Fallback if reward_metrics is not available
            total_rewards = batch["rewards"]
            format_rewards = [0.0] * len(total_rewards)
            correctness_rewards = total_rewards
        
        # Convert to tensors
        format_rewards = torch.tensor(format_rewards, dtype=torch.float, device=Q_logits.device)
        correctness_rewards = torch.tensor(correctness_rewards, dtype=torch.float, device=Q_logits.device)
        
        # Compute format reward logits using the format_reward_pred layer
        # Get penultimate layer activations (last hidden state before final layer norm)
        with torch.no_grad():
            outputs = policy_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
            )
            # Use the last hidden state (penultimate layer activations)
            penultimate_hidden = outputs.hidden_states[-1]  # [batch_size, seq_len, hidden_size]
        
        # Apply format reward prediction layer
        format_reward_logits = policy_model.format_reward_pred(penultimate_hidden)  # [batch_size, seq_len, 1]
        format_reward_logits = format_reward_logits.squeeze(-1)  # [batch_size, seq_len]
        
        # Shift to match the logps shape (remove last token)
        format_reward_logits = format_reward_logits[..., :-1]  # [batch_size, seq_len-1]
        
        # Update Q_logits to only use correctness rewards
        Q_logits = Q_logits - (1/algo_config["Q_beta"]) * algo_config.get("beta_format", 1.0) * format_reward_logits
        
        # Set up rewards for Q-loss (only correctness rewards)
        rewards = (correctness_rewards > 0).float()
    else:
        # Original behavior - use total rewards
        rewards = (torch.Tensor(batch["rewards"]).to(Q_logits.device) > 0).float()
        format_reward_logits = None

    assert rewards.shape == (Q_logits.shape[0],)

    rewards = rewards[:,None].expand(-1, Q_logits.shape[1])
    assert rewards.shape == Q_logits.shape

    # set rewards with masked locations to -100
    rewards = torch.where(labels_mask == 0, -100.0, rewards)
    
    # Compute Q-function loss (binary cross entropy on logits NOT masked by labels_mask)
    # Apply sigmoid to Q_logits to get probabilities
    Q_probs = torch.sigmoid(Q_logits)
    
    # Binary cross entropy loss
    # BCE loss = -[y * log(p) + (1-y) * log(1-p)]
    # where y is the target (rewards) and p is the predicted probability (Q_probs)
    bce_loss = -(rewards * torch.log(Q_probs + 1e-8) + (1 - rewards) * torch.log(1 - Q_probs + 1e-8))
    
    # Apply mask to ensure masked entries don't contribute to loss
    # Only compute loss where labels_mask > 0
    masked_bce_loss = bce_loss * labels_mask
    
    # Sum the loss and divide by the number of valid (non-masked) tokens
    Q_loss = masked_bce_loss.sum() / (labels_mask.sum() + 1e-8)

    # Compute format reward loss if split rewards is enabled
    format_loss = None
    if algo_config.get("split_rewards", False) and format_reward_logits is not None:
        # Set up format rewards tensor
        format_rewards_tensor = (format_rewards[:,None].expand(-1, format_reward_logits.shape[1]) > 0).float()
        format_rewards_tensor = torch.where(labels_mask == 0, -100.0, format_rewards_tensor)
        
        # Compute format reward probabilities
        format_reward_probs = torch.sigmoid(format_reward_logits)
        
        # Binary cross entropy loss for format rewards
        format_bce_loss = -(format_rewards_tensor * torch.log(format_reward_probs + 1e-8) + 
                           (1 - format_rewards_tensor) * torch.log(1 - format_reward_probs + 1e-8))
        
        # Apply mask and normalize
        masked_format_loss = format_bce_loss * labels_mask
        format_loss = masked_format_loss.sum() / (labels_mask.sum() + 1e-8)
        
        # Combine losses
        Q_loss = Q_loss + algo_config.get("beta_format", 1.0) * format_loss

    # Compute metrics
    metrics = {
        "Q_loss": Q_loss.item(),
    }
    
    # Add Q parameter values to metrics if they are learnable
    if hasattr(policy_model, 'Q_A') and isinstance(policy_model.Q_A, torch.nn.Parameter):
        metrics["Q_A"] = policy_model.Q_A.item()
    if hasattr(policy_model, 'Q_c') and isinstance(policy_model.Q_c, torch.nn.Parameter):
        metrics["Q_c"] = policy_model.Q_c.item()
    
    # Add format loss to metrics if split rewards is enabled
    if format_loss is not None:
        metrics["format_loss"] = format_loss.item()
        metrics["format_reward_logits_mean"] = format_reward_logits.mean().item()
        metrics["format_reward_logits_std"] = format_reward_logits.std().item()
    
    # Add logps and Q_logits statistics to metrics
    with torch.no_grad():
        metrics["policy_logps_mean"] = logps.mean().item()
        metrics["policy_logps_std"] = logps.std().item()
        metrics["ref_logps_mean"] = ref_logps.mean().item()
        metrics["ref_logps_std"] = ref_logps.std().item()
        metrics["Q_logits_mean"] = Q_logits.mean().item()
        metrics["Q_logits_std"] = Q_logits.std().item()
        metrics["logps_diff_mean"] = (logps - ref_logps).mean().item()
        metrics["logps_diff_std"] = (logps - ref_logps).std().item()

    return Q_loss, metrics



def compute_Q_loss(
    policy_model: Union[DeepSpeedEngine, PreTrainedModel],
    reference_model: Union[DeepSpeedEngine, PreTrainedModel],
    batch: Dict[str, torch.Tensor],
    total_response_len: int,
    TEMPERATURE: float,
    algo_config: Dict[str, Any] = None, 
    loss_type: str = "mc",
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute the Q-learning loss.
    """
    if loss_type == "mc":
        return compute_Q_mc_loss(policy_model, reference_model, batch, total_response_len, TEMPERATURE, algo_config)
    else:
        raise ValueError(f"Invalid loss type: {loss_type}")