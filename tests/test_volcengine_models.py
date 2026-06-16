import warnings

from utils.volcengine_models import JsonSchema, ResponseFormatJsonSchema


def test_json_schema_uses_alias_without_shadowing_basemodel_schema():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        json_schema = JsonSchema(
            name="reply",
            schema={"type": "object", "properties": {"message": {"type": "string"}}},
            strict=True,
        )

    assert not any("Field name \"schema\"" in str(item.message) for item in caught)
    assert callable(json_schema.schema)
    assert json_schema.schema_["type"] == "object"

    response_format = ResponseFormatJsonSchema(json_schema=json_schema)
    dumped = response_format.model_dump(by_alias=True)

    assert dumped["json_schema"]["schema"]["type"] == "object"
    assert "schema_" not in dumped["json_schema"]
