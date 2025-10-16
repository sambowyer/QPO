# QPO: Q-function approximation via model logits

## Conda environment with environment.yml

```bash
conda env create -f environment.yml
```

Environment name: `grpo`.

#### Bluepebble

N.B.: To get it working on bluepebble, I had to run

```
conda install -c nvidia cuda-compiler
```

after the regular env setup.


# Outline

So we're basically trying to train a network $f_\theta$ by optimising a Q-function that is defined using the model's logits $l_\theta(a|s)$ and the reference model's logits $l_{ref}(a|s)$.
Specifically, we treat the logits of a reward-classifier (by which we mean the logits of the Q-function) $l_{Q}$ as the weighted difference of these two logits

$$l_{Q}(a|s) = \frac{1}{\beta} (l_\theta(a|s) - l_{ref}(a|s)) + A l_{ref}(a|s) + c$$

where $\beta$ is a hyperparameter and $A$ and $c$ are learned parameters (maybe we'll just set $c=0$ and $A=1$).

## Training $Q$

We then have multiple options for how to train $Q$ (which happens implicitly in the training of $f_\theta$):

- **Monte Carlo:** Train $f_\theta$ to minimise the classification loss on the logits $l_Q$
- **Online Bootstrap:** Bellman updates $$Q(a_t | s_t) = \sum_a P_{\theta}(a | s(a_t, s_t))Q(a | s(a_t, s_t))$$ where $P_{\theta}$ is the policy.
- **Q-learning (Offline Bootstrap):** $$Q*(a_t|s_t) = \max_a Q*(a|s(a_t, s_t))$$

## Non-binary rewards
We can do some other stuff later on for this case, but first we need to check that binary rewards work.



# Implementation


Basically the training loop is:

```python
for batch in dataloader:
    # batch contains prompts, we need to rollout the model and get the logits
    rollouts = model.generate(batch, max_length=1024)

    logits = rollouts.logits
    logits_ref = ref_model(batch)

    logits_Q = (1/beta) * (logits - logits_ref) + A * logits_ref + c

    # Get the rewards (binary labels)
    rewards = get_rewards(rollouts)

    # Train the model
    loss = loss_fn(logits_Q, rewards)

    # Backprop
    loss.backward()
    optimizer.step()
```
