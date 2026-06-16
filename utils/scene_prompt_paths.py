"""Scene prompt path resolution shared by runtime and settings UI."""

from __future__ import annotations

from typing import Any, Dict, List


DEFAULT_SCENE_PROMPT_FILES: Dict[str, str] = {
    "presale": "runtime/scene_prompts_review/presale_prompt.txt",
    "insale": "runtime/scene_prompts_review/insale_prompt.txt",
    "aftersale": "runtime/scene_prompts_review/aftersale_prompt.txt",
}

LEGACY_SCENE_PROMPT_FILES: Dict[str, str] = {
    "presale": "runtime/scene_prompts_review/family_a_售前场景prompt_待审.txt",
    "insale": "runtime/scene_prompts_review/family_a_售中场景prompt_待审.txt",
    "aftersale": "runtime/scene_prompts_review/family_a_售后场景prompt_待审.txt",
}

_SCENE_ALIASES = {
    "presale": "presale",
    "pre_sale": "presale",
    "pre-sale": "presale",
    "售前": "presale",
    "售前咨询": "presale",
    "insale": "insale",
    "in_sale": "insale",
    "in-sale": "insale",
    "售中": "insale",
    "售中-待发货": "insale",
    "售中-物流中": "insale",
    "aftersale": "aftersale",
    "after_sale": "aftersale",
    "after-sale": "aftersale",
    "售后": "aftersale",
    "售后倾向": "aftersale",
}


def normalize_scene_prompt_key(scene: Any) -> str:
    clean = str(scene or "").strip().lower().replace(" ", "")
    if not clean:
        return ""
    direct = _SCENE_ALIASES.get(clean)
    if direct:
        return direct
    for alias, scene_key in _SCENE_ALIASES.items():
        if alias in clean:
            return scene_key
    return ""


def resolve_scene_prompt_files(configured: Any = None) -> Dict[str, str]:
    paths = dict(DEFAULT_SCENE_PROMPT_FILES)
    if not isinstance(configured, dict):
        return paths

    for scene, path in configured.items():
        scene_key = normalize_scene_prompt_key(scene)
        if not scene_key:
            continue
        path_text = str(path or "").strip()
        if path_text:
            paths[scene_key] = path_text
    return paths


def scene_prompt_read_candidates(scene_key: str, configured: Any = None) -> List[str]:
    scene = normalize_scene_prompt_key(scene_key)
    if not scene:
        return []

    paths = resolve_scene_prompt_files(configured)
    candidates = []
    for path in (paths.get(scene), LEGACY_SCENE_PROMPT_FILES.get(scene)):
        if path and path not in candidates:
            candidates.append(path)
    return candidates
