from utils.compound_id_utils import (
    build_compound_id_alias_map,
    normalize_compound_id_text,
    parse_compound_id_parts,
    resolve_compound_id_alias,
)


def check_hyphenated_intermediate_ids_are_preserved():
    assert normalize_compound_id_text("204-III") == "204-III"
    assert parse_compound_id_parts("204-III")["core"] == "204-III"
    assert parse_compound_id_parts("Compound 204-III")["core"] == "204-III"
    assert parse_compound_id_parts("Formula II-1")["core"] == "II-1"


def check_hyphenated_ids_do_not_collapse_to_parent_compound_number():
    alias_map, ambiguous = build_compound_id_alias_map(["Compound 204", "204-III"])

    assert resolve_compound_id_alias("204-III", alias_map, ambiguous) == "204-III"
    assert resolve_compound_id_alias("Compound 204", alias_map, ambiguous) == "Compound 204"
    assert resolve_compound_id_alias("204", alias_map, ambiguous) == "Compound 204"


def check_reaction_sub_ids_are_supported_as_exact_ids():
    alias_map, ambiguous = build_compound_id_alias_map(["44-1", "142-II", "222-IV"])

    assert resolve_compound_id_alias("44-1", alias_map, ambiguous) == "44-1"
    assert resolve_compound_id_alias("Compound 142-II", alias_map, ambiguous) == "142-II"
    assert resolve_compound_id_alias("222-IV", alias_map, ambiguous) == "222-IV"


def check_non_target_prefixes_do_not_alias_to_target_compounds():
    alias_map, ambiguous = build_compound_id_alias_map(["Compound 204"])

    assert resolve_compound_id_alias("204", alias_map, ambiguous) == "Compound 204"
    assert resolve_compound_id_alias("Formula 204", alias_map, ambiguous) == "Compound 204"
    assert resolve_compound_id_alias("Intermediate 204", alias_map, ambiguous) is None
    assert resolve_compound_id_alias("Int. 204", alias_map, ambiguous) is None
    assert resolve_compound_id_alias("Preparation 204", alias_map, ambiguous) is None
    assert resolve_compound_id_alias("Embodiment 204", alias_map, ambiguous) is None


def check_target_prefix_variants_alias_to_same_target_compound():
    alias_map, ambiguous = build_compound_id_alias_map(["Example 12"])

    assert resolve_compound_id_alias("Ex. 12", alias_map, ambiguous) == "Example 12"
    assert resolve_compound_id_alias("No. 12", alias_map, ambiguous) == "Example 12"
    assert resolve_compound_id_alias("Compound 12", alias_map, ambiguous) == "Example 12"
    assert resolve_compound_id_alias("编号12", alias_map, ambiguous) == "Example 12"
    assert resolve_compound_id_alias("Intermediate 12", alias_map, ambiguous) is None


if __name__ == "__main__":
    check_hyphenated_intermediate_ids_are_preserved()
    check_hyphenated_ids_do_not_collapse_to_parent_compound_number()
    check_reaction_sub_ids_are_supported_as_exact_ids()
    check_non_target_prefixes_do_not_alias_to_target_compounds()
    check_target_prefix_variants_alias_to_same_target_compound()
    print("compound id utility checks ok")
