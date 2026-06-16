from utils.transfer_target import build_transfer_candidates, choose_transfer_candidate


def test_build_transfer_candidates_ignores_non_mapping_cs_list():
    assert build_transfer_candidates("shop-1", "seller-1", ["cs-2"]) == []


def test_choose_transfer_candidate_ignores_non_mapping_cs_list():
    assert choose_transfer_candidate("shop-1", "seller-1", "bad payload") is None


def test_build_transfer_candidates_flattens_grouped_cs_list_without_using_group_key():
    candidates = build_transfer_candidates(
        "shop-1",
        "seller-1",
        {
            "mall_cs": [
                {"cs_uid": "cs_shop-1_seller-1", "username": "当前客服"},
                {"cs_uid": "cs_shop-1_seller-2", "username": "客服2"},
            ]
        },
    )

    assert candidates == [
        {
            "raw_cs_uid": "cs_shop-1_seller-2",
            "cs_uid": "cs_shop-1_seller-2",
            "username": "客服2",
            "info": {"cs_uid": "cs_shop-1_seller-2", "username": "客服2"},
        }
    ]
