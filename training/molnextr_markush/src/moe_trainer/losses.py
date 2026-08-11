"""Loss and reward functions for the MoE trainer.

Extracted from the original ``tools/train_moe.py`` monolith.
"""
from __future__ import annotations

from ._common import *  # noqa: F401,F403

def finite_metric(value) -> float | None:
    if value is None:
        return None
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return None
    return value_f if np.isfinite(value_f) else None


def fmt_metric(value) -> str:
    value_f = finite_metric(value)
    return "n/a" if value_f is None else f"{value_f:.3f}"


def group_relative_rloo_loss(
    sequence_log_probs: list[torch.Tensor],
    rewards: list[float],
) -> tuple[torch.Tensor | None, torch.Tensor]:
    """Unbiased leave-one-out group baseline without response-length scaling."""
    if len(sequence_log_probs) != len(rewards) or len(rewards) < 2:
        raise ValueError("group-relative RLOO requires matching groups of at least two")
    reward_tensor = torch.tensor(
        rewards,
        dtype=torch.float32,
        device=sequence_log_probs[0].device,
    )
    baseline = (reward_tensor.sum() - reward_tensor) / (len(rewards) - 1)
    advantages = reward_tensor - baseline
    if float(advantages.abs().max().item()) <= 1.0e-8:
        return None, advantages
    loss = torch.stack(
        [
            -advantage.detach() * log_prob
            for advantage, log_prob in zip(advantages, sequence_log_probs)
        ]
    ).mean()
    return loss, advantages


def decoded_prediction_smiles(prediction: dict) -> str:
    if prediction.get("decode_quality_issue"):
        return ""
    atom_data = prediction.get(ATOM_FORMAT, {})
    decoded_smiles, _, _, graph_issues = convert_graph_to_smiles(
        [atom_data.get("coords") or []],
        [atom_data.get("symbols") or []],
        [prediction.get("edges") or []],
        num_workers=1,
    )
    if graph_issues and graph_issues[0]:
        return ""
    return decoded_smiles[0] if decoded_smiles else ""


def reward_weighted_counterfactual_dpo_loss(
    alternative_log_prob: torch.Tensor,
    greedy_log_prob: torch.Tensor,
    reward_delta: float,
    *,
    beta: float,
    reward_margin: float,
) -> torch.Tensor | None:
    if abs(float(reward_delta)) < float(reward_margin):
        return None
    direction = 1.0 if float(reward_delta) > 0.0 else -1.0
    preference_logit = (
        direction
        * float(beta)
        * (alternative_log_prob - greedy_log_prob)
    )
    return abs(float(reward_delta)) * F.softplus(-preference_logit)


def searched_terminal_action_loss(
    terminal_action_logits: torch.Tensor | None,
    reward_gain: float,
    *,
    reward_margin: float,
    target_logit_margin: float,
) -> torch.Tensor | None:
    """Distill a searched bonded-dummy action at the deployed EOS prefix.

    The rollout exposes exactly two logits in ``[EOS, terminal dummy]`` order.
    Search, rather than the forced token identity, decides whether the update is
    admissible.  Once the complete graph reward clears the margin, a pairwise
    large-margin logistic objective gives the boundary action direct credit
    instead of diluting it across the full alternative completion.
    """
    if float(reward_gain) < float(reward_margin):
        return None
    if (
        not isinstance(terminal_action_logits, torch.Tensor)
        or terminal_action_logits.numel() != 2
    ):
        return None
    return terminal_action_pair_margin_loss(
        terminal_action_logits,
        target_star=True,
        target_logit_margin=target_logit_margin,
    )


def terminal_action_pair_margin_loss(
    terminal_action_logits: torch.Tensor | None,
    *,
    target_star: bool,
    target_logit_margin: float,
) -> torch.Tensor | None:
    if (
        not isinstance(terminal_action_logits, torch.Tensor)
        or terminal_action_logits.numel() != 2
    ):
        return None
    pair = terminal_action_logits.reshape(2).float()
    if not bool(torch.isfinite(pair).all().detach().item()):
        return None
    preferred_margin = pair[1] - pair[0]
    if not bool(target_star):
        preferred_margin = -preferred_margin
    return F.softplus(float(target_logit_margin) - preferred_margin)


def terminal_dummy_edge_expected_reward(
    prediction: dict,
    gold_smiles: str,
    *,
    distill_best_bonded_action: bool = False,
    min_reward_gain: float | None = None,
) -> tuple[torch.Tensor | None, dict | None, dict]:
    """Exactly marginalize the terminal dummy's no-bond/anchor/bond actions."""
    atom_data = prediction.get(ATOM_FORMAT, {})
    symbols = list(atom_data.get("symbols") or [])
    coords = list(atom_data.get("coords") or [])
    edge_logits = prediction.get("_rollout_edge_logits")
    dummy_indices = [
        index for index, symbol in enumerate(symbols) if "*" in str(symbol)
    ]
    if (
        len(dummy_indices) != 1
        or not isinstance(edge_logits, torch.Tensor)
        or edge_logits.ndim != 4
        or edge_logits.size(0) != 1
        or edge_logits.size(1) != 7
        or edge_logits.size(2) != len(symbols)
        or edge_logits.size(3) != len(symbols)
    ):
        return None, None, {"candidate_count": 0, "reason": "missing_terminal_dummy_logits"}
    dummy_index = dummy_indices[0]
    anchors = [index for index in range(len(symbols)) if index != dummy_index]
    if not anchors:
        return None, None, {"candidate_count": 0, "reason": "missing_anchor"}

    directed = F.softmax(edge_logits.float()[0], dim=0).permute(1, 2, 0)
    pair_probabilities = []
    for anchor_index in anchors:
        forward = directed[dummy_index, anchor_index]
        reverse = directed[anchor_index, dummy_index]
        pair_probabilities.append(
            torch.stack(
                [
                    *[
                        0.5 * (forward[bond_type] + reverse[bond_type])
                        for bond_type in range(5)
                    ],
                    0.5 * (forward[5] + reverse[6]),
                    0.5 * (forward[6] + reverse[5]),
                ]
            ).clamp_min(1.0e-8)
        )
    pair_probabilities = torch.stack(pair_probabilities)
    no_bond_score = pair_probabilities[:, 0].log().sum()
    actions: list[tuple[int, int]] = [(-1, 0)]
    action_scores = [no_bond_score]
    for local_anchor, anchor_index in enumerate(anchors):
        bond_type = 1
        actions.append((anchor_index, bond_type))
        action_scores.append(
            no_bond_score
            - pair_probabilities[local_anchor, 0].log()
            + pair_probabilities[local_anchor, bond_type].log()
        )
    action_scores_tensor = torch.stack(action_scores)
    action_probabilities = F.softmax(action_scores_tensor, dim=0)

    raw_edges = prediction.get("edges") or []
    if len(raw_edges) != len(symbols) or any(
        len(row) != len(symbols) for row in raw_edges
    ):
        raw_edges = [[0] * len(symbols) for _ in symbols]
    candidate_edges = []
    for anchor_index, bond_type in actions:
        edges = [list(row) for row in raw_edges]
        for index in range(len(symbols)):
            edges[dummy_index][index] = 0
            edges[index][dummy_index] = 0
        if anchor_index >= 0:
            edges[dummy_index][anchor_index] = bond_type
            edges[anchor_index][dummy_index] = (
                6 if bond_type == 5 else 5 if bond_type == 6 else bond_type
            )
        candidate_edges.append(edges)

    decoded_smiles, _, _, graph_issues = convert_graph_to_smiles(
        [coords for _ in actions],
        [symbols for _ in actions],
        candidate_edges,
        num_workers=1,
    )
    metrics = [
        fragment_graph_reward(
            "" if issue else smiles,
            gold_smiles,
        )
        for smiles, issue in zip(decoded_smiles, graph_issues)
    ]
    rewards = torch.tensor(
        [float(item["reward"]) for item in metrics],
        dtype=torch.float32,
        device=action_scores_tensor.device,
    )
    best_index = int(torch.argmax(rewards).item())
    best_bonded_index = int(torch.argmax(rewards[1:]).item()) + 1
    reward_range = float((rewards.max() - rewards.min()).item())
    best_bonded_gain = float(
        (rewards[best_bonded_index] - rewards[0]).item()
    )
    audit = {
        "candidate_count": len(actions),
        "reward_range": reward_range,
        "no_bond_reward": float(rewards[0].item()),
        "best_reward": float(rewards[best_index].item()),
        "best_anchor": int(actions[best_index][0]),
        "best_bond_type": int(actions[best_index][1]),
        "best_bonded_reward": float(rewards[best_bonded_index].item()),
        "best_bonded_reward_gain": best_bonded_gain,
        "best_bonded_anchor": int(actions[best_bonded_index][0]),
        "best_bonded_bond_type": int(actions[best_bonded_index][1]),
        "expected_reward": float(
            (action_probabilities.detach() * rewards).sum().item()
        ),
    }
    if distill_best_bonded_action:
        search_policy_applied = bool(
            min_reward_gain is None
            or best_bonded_gain >= float(min_reward_gain)
        )
        audit["search_policy_target_probability"] = float(
            action_probabilities[best_bonded_index].detach().item()
        )
        audit["search_policy_applied"] = search_policy_applied
        if not search_policy_applied:
            return None, metrics[best_bonded_index], audit
        search_policy_loss = -F.log_softmax(
            action_scores_tensor.float(), dim=0
        )[best_bonded_index]
        return search_policy_loss, metrics[best_bonded_index], audit
    if reward_range <= 1.0e-8:
        return None, metrics[best_index], audit
    expected_advantage = (
        action_probabilities * (rewards - rewards[0])
    ).sum()
    return -expected_advantage, metrics[best_index], audit


