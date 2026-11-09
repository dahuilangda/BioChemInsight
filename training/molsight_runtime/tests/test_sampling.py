from collections import Counter

from molsight.sampling import DomainBatchSampler


def metadata():
    return [
        {"category": category, "peptide_stratum": stratum, "input_size": resolution}
        for category, stratum in [
            ("complete", None),
            ("peptide", "linear"),
            ("peptide", "bicyclic"),
        ]
        for resolution in (512, 768)
        for _ in range(40)
    ]


def sampler(rows, **overrides):
    return DomainBatchSampler(
        rows,
        updates=500,
        accumulation=3,
        batch_size=2,
        weights={"complete": 0.8, "peptide": 0.2},
        **overrides,
    )


def test_distributed_microbatches_share_resolution_without_rank_overlap():
    rows = metadata()
    first, second = (
        sampler(rows, rank=0, world_size=2),
        sampler(rows, rank=1, world_size=2),
    )
    assert first.definition_sha256 == second.definition_sha256
    for left, right in zip(first, second, strict=True):
        assert not set(left) & set(right)
        selected = [rows[index] for index in left + right]
        assert len({row["input_size"] for row in selected}) == 1
        assert len({row["category"] for row in selected}) == 1


def test_domains_are_weighted_per_optimizer_update_and_resume_exactly():
    rows = metadata()
    batches = list(sampler(rows))
    counts = Counter(
        rows[batches[index][0]]["category"] for index in range(0, len(batches), 3)
    )
    assert 0.15 < counts["peptide"] / 500 < 0.25
    for start in range(0, len(batches), 3):
        assert (
            len(
                {
                    rows[index]["category"]
                    for batch in batches[start : start + 3]
                    for index in batch
                }
            )
            == 1
        )
    resumed = list(sampler(rows, start_update=200))
    assert resumed == batches[200 * 3 :]


def test_discovered_peptides_remain_in_their_original_recognition_domain():
    rows = [
        {
            "category": "complete",
            "input_size": 512,
            "peptide_candidate": True,
            "cyclic_peptide_candidate": True,
        },
        {"category": "complete", "input_size": 512},
    ]
    sampling = sampler(rows)
    assert sampling.groups["complete"][""][512].tolist() == [0, 1]
    assert sampling.groups["peptide"]["discovered_cyclic"][512].tolist() == [0]
    assert rows[0]["category"] == "complete"


def test_excluded_domains_cannot_enter_through_peptide_tags():
    rows = metadata() + [
        {
            "category": "markush",
            "input_size": 512,
            "peptide_candidate": True,
            "cyclic_peptide_candidate": True,
        }
    ]
    sampling = sampler(rows, excluded_domains=["markush"])
    for batch in sampling:
        assert len(rows) - 1 not in batch
    for update, batch in enumerate(list(sampling)[::3]):
        assert rows[batch[0]]["category"] == sampling.domain_for_update(update)


def test_peptide_groups_are_weighted_before_synthetic_strata():
    rows = metadata() + [
        {
            "category": "complete",
            "input_size": 512,
            "peptide_candidate": True,
            "cyclic_peptide_candidate": cyclic,
        }
        for cyclic in (False, True)
        for _ in range(40)
    ]
    weights = {"synthetic": 0.5, "discovered_cyclic": 0.35, "discovered_linear": 0.15}
    first = sampler(rows, world_size=2, peptide_group_weights=weights)
    second = sampler(rows, rank=1, world_size=2, peptide_group_weights=weights)
    batches = list(first)
    counts = Counter()
    for update in range(500):
        if first.domain_for_update(update) == "peptide":
            row = rows[batches[update * 3][0]]
            group = (
                "synthetic"
                if row.get("peptide_stratum")
                else "discovered_cyclic"
                if row.get("cyclic_peptide_candidate")
                else "discovered_linear"
            )
            counts[group] += 1
    total = counts.total()
    assert 0.35 < counts["synthetic"] / total < 0.65
    assert 0.2 < counts["discovered_cyclic"] / total < 0.5
    assert 0.05 < counts["discovered_linear"] / total < 0.3
    assert (
        list(
            sampler(rows, world_size=2, start_update=200, peptide_group_weights=weights)
        )
        == batches[600:]
    )
    assert all(
        not set(left) & set(right) for left, right in zip(batches, second, strict=True)
    )
    assert first.definition_sha256 != sampler(rows, world_size=2).definition_sha256


def test_peptide_group_weights_reject_missing_or_invalid_groups():
    import pytest

    with pytest.raises(ValueError, match="exactly"):
        sampler(metadata(), peptide_group_weights={"discovered_cyclic": 1.0})
    with pytest.raises(ValueError, match="positive"):
        sampler(metadata(), peptide_group_weights={"synthetic": float("nan")})


def test_stratum_weights_concentrate_and_validate():
    rows = metadata()
    rows += [
        {"category": "peptide", "peptide_stratum": "patent", "input_size": 512}
        for _ in range(20)
    ]
    weighted = sampler(
        rows,
        stratum_weights={"patent": 6.0, "bicyclic": 1.0},
    )
    seen = Counter()
    for batch in weighted:
        if rows[batch[0]]["category"] == "peptide":
            seen[rows[batch[0]]["peptide_stratum"]] += 1
    total = sum(seen.values())
    assert total > 0
    assert seen["patent"] / total > 0.5, "weighted stratum must dominate"
    assert seen["linear"] > 0, "unlisted strata keep weight 1 and stay present"
    try:
        sampler(rows, stratum_weights={"nonexistent": 2.0})
    except ValueError as error:
        assert "absent from the corpus" in str(error)
    else:
        raise AssertionError("unknown strata must raise")
