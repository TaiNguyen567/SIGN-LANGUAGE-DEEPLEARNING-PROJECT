from src.data.tokenizer import SignTokenizer


def test_unicode_vietnamese_round_trip(tmp_path):
    text = "Tôi muốn uống nước."
    tokenizer = SignTokenizer.build([text, "ĂN CƠM", "ĐI HỌC"])

    ids = tokenizer.encode(text)

    assert tokenizer.decode(ids) == "TÔI MUỐN UỐNG NƯỚC."
    path = tmp_path / "vocab.json"
    tokenizer.save(path)
    assert SignTokenizer.load(path).decode(ids) == "TÔI MUỐN UỐNG NƯỚC."


def test_unknown_and_blank_tokens_are_handled():
    tokenizer = SignTokenizer.build(["XIN CHÀO"])

    assert tokenizer.encode("TÔI") == [tokenizer.unknown_id]
    assert tokenizer.decode([0, *tokenizer.encode("XIN CHÀO"), 0]) == "XIN CHÀO"


def test_atomic_class_token_decodes_to_display_label():
    tokenizer = SignTokenizer.build(["Bệnh_nhân"])

    assert tokenizer.encode("Bệnh_nhân") == [2]
    assert tokenizer.decode([2]) == "BỆNH NHÂN"
