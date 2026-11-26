"""
Direct Preference Optimization (DPO) from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - log_softmax
def log_softmax(logits, axis=-1):
    # TODO: convert logits into numerically stable log-probabilities along axis
    x = np.asarray(logits, dtype=np.float64)
    max_x = np.max(x, axis=axis, keepdims=True)
    shifted = x - max_x
    log_sum_exp = np.log(
        np.sum(np.exp(shifted), axis=axis, keepdims=True)
    )
    return shifted - log_sum_exp

# Step 2 - softmax
def softmax(logits, axis=-1):
    # TODO: Convert an array of logits into a probability distribution along a given axis
    x = np.asarray(logits, dtype=np.float64)
    max_x = np.max(x, axis=axis, keepdims=True)
    shifted = x - max_x
    exp_x = np.exp(shifted)
    return exp_x / np.sum(
        exp_x, axis=axis, keepdims=True
    )

# Step 3 - gather_token_logprobs
def gather_token_logprobs(log_probs, token_ids):
    # TODO: Extract the log-probability of each observed token from a full vocab log-prob tensor...
    B, T, V = log_probs.shape
    batch_idx = np.arange(B)[:, None]
    time_idx = np.arange(T)[None, :]
    return log_probs[batch_idx, time_idx, token_ids]

# Step 4 - masked_sequence_logprob
def masked_sequence_logprob(token_logprobs, mask):
    # TODO: Sum per-token log-probabilities under a binary mask to obtain a single sequence log-probability per example.
    return np.sum(token_logprobs * mask, axis=1)

# Step 5 - init_policy_params
def init_policy_params(vocab_size, d_model, rng=None):
    # TODO: Initialize the policy language-model parameters with small random values
    if rng is None:
        rng = np.random.default_rng()
    return {
        "embed": rng.normal(
            0.0, 0.02, size=(vocab_size, d_model)
        ),
        "W_out": rng.normal(
            0.0, 0.02, size=(d_model, vocab_size)
        ),
        "b_out": np.zeros(vocab_size, dtype=float)
    }

# Step 6 - policy_token_logits
def policy_token_logits(params, token_ids):
    # TODO: Compute next-token logits for every position from policy params and token ids.
    embeddings = params["embed"][token_ids]
    logits = embeddings @ params["W_out"] + params["b_out"]
    return logits

# Step 7 - policy_sequence_logprob
def policy_sequence_logprob(params, token_ids, mask):
    # TODO: Compute the total masked sequence log-probability under the current policy...
    logits = policy_token_logits(params, token_ids)
    log_probs = log_softmax(logits, axis=-1)
    token_logprobs = gather_token_logprobs(log_probs, token_ids)
    seq_logprobs = masked_sequence_logprob(token_logprobs, mask)
    return seq_logprobs

# Step 8 - sequence_logprob_grad
def sequence_logprob_grad(params, token_ids, mask):
    # TODO: Compute gradients of the summed sequence log-probability w.r.t. params
    logits = policy_token_logits(params, token_ids)
    probs = softmax(logits, axis=-1)

    B, T = token_ids.shape

    dlogits = -probs.copy()

    b_idx = np.arange(B)[:, None]
    t_idx = np.arange(T)[None, :]

    dlogits[b_idx, t_idx, token_ids] += 1.0
    dlogits *= mask[..., None]

    embeddings = params["embed"][token_ids]

    dW_out = np.einsum(
        "btd, btv->dv", embeddings, dlogits
    )
    db_out = np.einsum("btv->v", dlogits)
    dembeddings = np.einsum("btv, dv->btd", dlogits, params["W_out"])
    dembed = np.zeros_like(params["embed"])
    np.add.at(dembed, token_ids, dembeddings)

    return {
        "embed": dembed,
        "W_out": dW_out,
        "b_out": db_out
    }

# Step 9 - bradley_terry_loss
import math
from statistics import fmean

def bradley_terry_loss(reward_chosen, reward_rejected):
    # TODO: Compute the mean Bradley-Terry pairwise preference loss...
    return fmean(
        math.log1p(math.exp(-(c - r)))
        for c, r in zip(reward_chosen, reward_rejected)
    )

# Step 10 - reward_accuracy
def reward_accuracy(reward_chosen, reward_rejected):
    # TODO: Fraction of pairs where chosen reward is strictly higher than rejected.
    if reward_chosen.shape != reward_rejected.shape:
        raise ValueError(
            f"shape mismatch: {reward_chosen.shape} vs {reward_rejected.shape}"
        )
    if reward_chosen.size == 0:
        return 0.0
    
    return float(np.mean(reward_chosen > reward_rejected))

# Step 11 - build_preference_pairs
def build_preference_pairs(prompts, chosen_ids, rejected_ids, chosen_mask, rejected_mask):
    # TODO: Package raw arrays into a list of preference-pair dictionaries
    prompts = list(prompts)
    n = len(prompts)

    for name, row in (('chosen_ids', chosen_ids),
                        ('rejected_ids', rejected_ids),
                        ('chosen_mask', chosen_mask),
                        ('rejected_mask', rejected_mask)):
        if len(row) != n:
            raise ValueError(f"{name} has {len(row)} rows, expected {n}")
    
    return [
        {
            'prompt': prompts[i],
            'chosen_ids': chosen_ids[i],
            'rejected_ids': rejected_ids[i],
            'chosen_mask': chosen_mask[i],
            'rejected_mask': rejected_mask[i],
        }
        for i in range(n)
    ]

# Step 12 - sample_preference_batch
_SEQ_FIELDS = ('chosen_ids', 'rejected_ids', 'chosen_mask', 'rejected_mask')



def sample_preference_batch(pairs, batch_size, rng=None):
    # TODO: Sample a mini-batch of preference pairs for one training step.
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not pairs:
        raise ValueError("pairs must not be empty")
    
    if rng is None:
        rng = np.random.default_rng()
    indices = rng.choice(
        len(pairs),
        size=batch_size,
        replace=batch_size > len(pairs),
    )

    sampled = [pairs[i] for i in indices]

    fields = (
        "chosen_ids",
        "rejected_ids",
        "chosen_mask",
        "rejected_mask",
    )

    batch = {
        field: np.stack([pair[field] for pair in sampled], axis=0)
        for field in fields
    }

    if any("prompt" in pair for pair in sampled):
        if all("prompt" in pair for pair in sampled):
            batch["prompt"] = np.stack(
                [pair["prompt"] for pair in sampled], axis=0
            )
        else:
            prompts = np.empty(batch_size, dtype=object)
            prompts[:] = [pair.get("prompt") for pair in sampled]
            batch["prompt"] = prompts
    return batch

# Step 13 - freeze_reference_logprobs
import numpy as np

def freeze_reference_logprobs(ref_params, pairs):
    """Precompute reference log-probs for all preference pairs."""
    frozen = []

    for pair in pairs:
        chosen_ids = np.asarray(pair["chosen_ids"])[None, :]
        chosen_mask = np.asarray(pair["chosen_mask"])[None, :]

        rejected_ids = np.asarray(pair["rejected_ids"])[None, :]
        rejected_mask = np.asarray(pair["rejected_mask"])[None, :]

        chosen_logprob = policy_sequence_logprob(
            ref_params,
            chosen_ids,
            chosen_mask,
        )

        rejected_logprob = policy_sequence_logprob(
            ref_params,
            rejected_ids,
            rejected_mask,
        )

        frozen.append({
            "chosen": float(np.asarray(chosen_logprob).item()),
            "rejected": float(np.asarray(rejected_logprob).item()),
        })

    return frozen

# Step 14 - policy_reference_logratio
def policy_reference_logratio(policy_logprob, reference_logprob):
    # TODO: Compute the per-sequence log-ratio log pi_theta(y) - log pi_ref(y)
    return policy_logprob - reference_logprob

# Step 15 - dpo_pair_margin
def dpo_pair_margin(policy_logprob_chosen, policy_logprob_rejected, ref_logprob_chosen, ref_logprob_rejected, beta):
    # TODO: Compute the scaled DPO pair margin for a batch of preference pairs
    if beta <= 0:
        raise ValueError("beta must be positive")
    chosen_ratio = policy_reference_logratio(
        policy_logprob_chosen,
        ref_logprob_chosen,
    )
    rejected_ratio = policy_reference_logratio(
        policy_logprob_rejected,
        ref_logprob_rejected
    )

    margin = beta * (chosen_ratio - rejected_ratio)

    return margin

# Step 16 - dpo_loss
def dpo_loss(policy_logprob_chosen, policy_logprob_rejected, ref_logprob_chosen, ref_logprob_rejected, beta):
    # TODO: return the mean logistic loss on the DPO pair margins as a scalar float
    margins = dpo_pair_margin(
        policy_logprob_chosen,
        policy_logprob_rejected,
        ref_logprob_chosen,
        ref_logprob_rejected,
        beta,
    )

    losses = np.logaddexp(0, -margins)
    return float(np.mean(losses))

# Step 17 - dpo_loss_grad (not yet solved)
# TODO: implement

# Step 18 - dpo_train_step (not yet solved)
# TODO: implement

# Step 19 - train_dpo (not yet solved)
# TODO: implement

# Step 20 - length_normalized_logprob (not yet solved)
# TODO: implement

# Step 21 - ipo_loss (not yet solved)
# TODO: implement

# Step 22 - implicit_reward (not yet solved)
# TODO: implement

# Step 23 - preference_accuracy (not yet solved)
# TODO: implement

# Step 24 - kl_to_reference (not yet solved)
# TODO: implement

# Step 25 - reward_margin_stats (not yet solved)
# TODO: implement

# Step 26 - evaluate_dpo (not yet solved)
# TODO: implement

# Step 27 - run_dpo_pipeline (not yet solved)
# TODO: implement

