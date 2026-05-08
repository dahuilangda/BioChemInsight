from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Mapping
import json
from copy import deepcopy


REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_SKILLS_ROOT = REPO_ROOT / "model_skills"


def _strip_frontmatter(text: str) -> str:
    if not text.startswith("---\n"):
        return text.strip()

    end_marker = "\n---\n"
    end_index = text.find(end_marker, 4)
    if end_index == -1:
        return text.strip()
    return text[end_index + len(end_marker):].strip()


@lru_cache(maxsize=None)
def load_skill_body(skill_name: str) -> str:
    skill_path = MODEL_SKILLS_ROOT / skill_name / "SKILL.md"
    if not skill_path.exists():
        raise FileNotFoundError(f"Skill file not found: {skill_path}")
    return _strip_frontmatter(skill_path.read_text(encoding="utf-8"))


@lru_cache(maxsize=None)
def load_skill_reference(skill_name: str, relative_path: str) -> str:
    skill_root = MODEL_SKILLS_ROOT / skill_name
    reference_path = skill_root / relative_path
    if not reference_path.exists():
        raise FileNotFoundError(f"Skill reference not found: {reference_path}")
    return reference_path.read_text(encoding="utf-8").strip()


def render_skill_reference(
    skill_name: str,
    relative_path: str,
    variables: Mapping[str, object] | None = None,
) -> str:
    # Ensure the skill exists and follows the SKILL.md convention, even if the
    # runtime prompt is loaded from a smaller reference file.
    load_skill_body(skill_name)
    content = load_skill_reference(skill_name, relative_path)
    for key, value in (variables or {}).items():
        content = content.replace(f"{{{{{key}}}}}", "" if value is None else str(value))
    return content.strip()


def render_skill_prompt_with_examples(
    skill_name: str,
    prompt_relative_path: str,
    examples_relative_path: str | None = None,
    variables: Mapping[str, object] | None = None,
) -> str:
    prompt = render_skill_reference(skill_name, prompt_relative_path, variables)
    if not examples_relative_path:
        return prompt
    examples = render_skill_reference(skill_name, examples_relative_path, variables)
    if not examples:
        return prompt
    return f"{prompt}\n\nFew-shot examples\n{examples}".strip()


@lru_cache(maxsize=None)
def load_skill_json(skill_name: str, relative_path: str) -> dict:
    skill_root = MODEL_SKILLS_ROOT / skill_name
    reference_path = skill_root / relative_path
    if not reference_path.exists():
        raise FileNotFoundError(f"Skill JSON reference not found: {reference_path}")
    return json.loads(reference_path.read_text(encoding="utf-8"))


def _deep_merge_dicts(base: dict, override: dict) -> dict:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge_dicts(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


@lru_cache(maxsize=None)
def load_merged_skill_json(
    base_skill_name: str,
    base_relative_path: str,
    override_skill_name: str,
    override_relative_path: str,
) -> dict:
    base = load_skill_json(base_skill_name, base_relative_path)
    override = load_skill_json(override_skill_name, override_relative_path)
    if not isinstance(base, dict) or not isinstance(override, dict):
        raise ValueError("Merged skill JSON inputs must both be JSON objects.")
    return _deep_merge_dicts(base, override)
