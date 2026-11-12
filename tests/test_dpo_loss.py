import torch

from align.dpo import dpo_loss


def test_loss_is_low_when_policy_prefers_chosen_more_than_reference_did():
    # policy pulls chosen up and rejected down relative to the frozen reference
    policy_chosen = torch.tensor(-5.0)
    policy_rejected = torch.tensor(-9.0)
    ref_chosen = torch.tensor(-6.0)
    ref_rejected = torch.tensor(-6.0)
    loss, margin, acc = dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=0.1)
    assert loss.item() < 0.6931  # better than the -log(0.5) loss at the decision boundary
    assert margin.item() > 0
    assert acc.item() == 1.0


def test_loss_is_high_when_policy_prefers_rejected():
    policy_chosen = torch.tensor(-9.0)
    policy_rejected = torch.tensor(-5.0)
    ref_chosen = torch.tensor(-6.0)
    ref_rejected = torch.tensor(-6.0)
    loss, margin, acc = dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=0.1)
    assert loss.item() > 0.6931
    assert margin.item() < 0
    assert acc.item() == 0.0


def test_loss_is_exactly_log2_when_policy_equals_reference():
    # no drift from the reference at all -> logits are exactly 0 -> -log(sigmoid(0)) = log 2
    same = torch.tensor(-6.0)
    loss, margin, _ = dpo_loss(same, same, same, same, beta=0.5)
    assert abs(loss.item() - 0.6931471805599453) < 1e-5
    assert margin.item() == 0.0


def test_higher_beta_amplifies_the_same_drift():
    policy_chosen, policy_rejected = torch.tensor(-5.0), torch.tensor(-8.0)
    ref_chosen, ref_rejected = torch.tensor(-6.0), torch.tensor(-6.0)
    loss_low, _, _ = dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=0.05)
    loss_high, _, _ = dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=1.0)
    assert loss_high.item() < loss_low.item()  # same drift counts for more at high beta
