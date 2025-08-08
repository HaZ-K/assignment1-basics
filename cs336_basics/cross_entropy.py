import torch

def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    # logits: [..., vocab_size]
    # targets: [...], same shape as logits.shape[:-1]

    # Step 1: Subtract max for numerical stability
    logits_stable = logits - logits.max(dim=-1, keepdim=True).values

    # Step 2: Compute logsumexp
    logsumexp = torch.logsumexp(logits_stable, dim=-1)
    # logsumexp = torch.log(torch.exp(logits_stable).sum(dim=-1))  # shape: [...]

    # Step 3: Gather the logit values at the target indices
    # logits_stable shape: [..., vocab_size]
    # targets shape: [...]
    # Use gather to index target values
    target_logits = logits_stable.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)  # shape: [...]

    # Step 4: Cross entropy loss
    loss = -target_logits + logsumexp  # shape: [...]

    # Step 5: Average over all elements (mean loss over batch)
    return loss.mean()
