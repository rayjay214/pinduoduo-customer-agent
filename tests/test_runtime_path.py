from utils.runtime_path import adjust_config_for_runtime, get_temp_path


def test_adjust_config_for_runtime_uses_temp_path_for_relative_paths():
    adjusted = adjust_config_for_runtime({"cache_path": "cache.sqlite"})

    assert adjusted["cache_path"] == str(get_temp_path("cache.sqlite"))


def test_adjust_config_for_runtime_does_not_mutate_nested_input():
    original = {
        "knowledge_base": {
            "contents_db_path": "contents.db",
            "vector_db_path": "vector_db",
        }
    }

    adjusted = adjust_config_for_runtime(original)

    assert original["knowledge_base"]["contents_db_path"] == "contents.db"
    assert original["knowledge_base"]["vector_db_path"] == "vector_db"
    assert adjusted["knowledge_base"]["contents_db_path"] == str(get_temp_path("contents.db"))
