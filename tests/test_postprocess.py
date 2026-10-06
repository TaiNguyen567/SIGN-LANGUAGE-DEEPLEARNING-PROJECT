from src.inference.postprocess import confidence_filter, merge_predictions, remove_duplicate_tokens, temporal_smoothing
from src.translation.vietnamese import VietnamesePostProcessor


def test_overlapping_predictions_are_merged_once():
    assert merge_predictions(["XIN", "CHÀO"], ["CHÀO", "TÔI", "TÊN"]) == ["XIN", "CHÀO", "TÔI", "TÊN"]
    assert remove_duplicate_tokens(["XIN", "XIN", "CHÀO"]) == ["XIN", "CHÀO"]


def test_smoothing_and_confidence_filter():
    hypotheses = [["TÔI"], ["TÔI", "MUỐN"], ["TÔI", "MUỐN"]]
    assert temporal_smoothing(hypotheses, min_votes=2) == ["TÔI", "MUỐN"]
    assert confidence_filter(["TÔI"], 0.4, 0.7) == []


def test_vietnamese_postprocessing_is_conservative():
    postprocessor = VietnamesePostProcessor()
    assert postprocessor.translate(["TÔI", "TÊN", "NAM"]) == "Tôi tên là Nam."
    assert postprocessor.translate(["CƠM", "ĂN"]) == "Cơm ăn."
    assert postprocessor.translate(["CẢM_ƠN"]) == "Cảm ơn."
    assert postprocessor.translate("XIN CHÀO!") == "Xin chào!"


def test_vietnamese_postprocessor_accepts_custom_phrase_reordering_rules():
    postprocessor = VietnamesePostProcessor({("CƠM", "ĂN"): ("ĂN", "CƠM")})

    assert postprocessor.translate(["CƠM", "ĂN"]) == "Ăn cơm."

    configured = VietnamesePostProcessor.from_config({"CƠM ĂN": "ĂN CƠM"})
    assert configured.translate(["CƠM", "ĂN"]) == "Ăn cơm."


def test_unavailable_rules_are_detected_against_atomic_model_labels():
    postprocessor = VietnamesePostProcessor.from_config({"CƠM ĂN": "ĂN CƠM"})
    vocabulary = ["<blank>", "<unk>", "THỨC_ĂN", "ĂN_MỪNG"]

    assert postprocessor.unavailable_rules(vocabulary) == [("CƠM", "ĂN")]


def test_rule_validation_accepts_atomic_multiword_classes():
    postprocessor = VietnamesePostProcessor.from_config({"CẢM ƠN": "CẢM ƠN"})

    assert postprocessor.unavailable_rules(["<blank>", "<unk>", "CẢM_ƠN"]) == []
