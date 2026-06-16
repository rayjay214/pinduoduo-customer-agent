from utils.scene_prompt_paths import (
    DEFAULT_SCENE_PROMPT_FILES,
    LEGACY_SCENE_PROMPT_FILES,
    normalize_scene_prompt_key,
    resolve_scene_prompt_files,
    scene_prompt_read_candidates,
)


def test_normalize_scene_prompt_key_accepts_common_aliases():
    assert normalize_scene_prompt_key("售前") == "presale"
    assert normalize_scene_prompt_key("pre-sale") == "presale"
    assert normalize_scene_prompt_key("售中-物流中") == "insale"
    assert normalize_scene_prompt_key("after_sale") == "aftersale"
    assert normalize_scene_prompt_key("unknown") == ""


def test_resolve_scene_prompt_files_uses_defaults_for_missing_scenes():
    paths = resolve_scene_prompt_files({"售前": "runtime/prompts/custom-presale.txt"})

    assert paths["presale"] == "runtime/prompts/custom-presale.txt"
    assert paths["insale"] == DEFAULT_SCENE_PROMPT_FILES["insale"]
    assert paths["aftersale"] == DEFAULT_SCENE_PROMPT_FILES["aftersale"]


def test_resolve_scene_prompt_files_ignores_unknown_or_empty_entries():
    paths = resolve_scene_prompt_files(
        {
            "presale": "",
            "unknown": "runtime/prompts/ignored.txt",
            "售后": "runtime/prompts/custom-aftersale.txt",
        }
    )

    assert paths["presale"] == DEFAULT_SCENE_PROMPT_FILES["presale"]
    assert paths["aftersale"] == "runtime/prompts/custom-aftersale.txt"
    assert "unknown" not in paths


def test_scene_prompt_read_candidates_include_legacy_fallback_for_defaults():
    candidates = scene_prompt_read_candidates("售前")

    assert candidates[0] == DEFAULT_SCENE_PROMPT_FILES["presale"]
    assert LEGACY_SCENE_PROMPT_FILES["presale"] in candidates


def test_scene_prompt_read_candidates_keep_configured_path_first():
    candidates = scene_prompt_read_candidates("presale", {"presale": "runtime/prompts/shop-a.txt"})

    assert candidates[0] == "runtime/prompts/shop-a.txt"
    assert LEGACY_SCENE_PROMPT_FILES["presale"] in candidates
