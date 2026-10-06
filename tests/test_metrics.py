from src.evaluation.metrics import character_error_rate, sentence_accuracy, word_error_rate


def test_error_metrics_keep_vietnamese_unicode():
    references = ["TÔI UỐNG NƯỚC", "XIN CHÀO"]
    hypotheses = ["TÔI UỐNG NƯỚC", "XIN CHÀO"]

    assert character_error_rate(references, hypotheses) == 0.0
    assert word_error_rate(references, hypotheses) == 0.0
    assert sentence_accuracy(references, hypotheses) == 1.0


def test_word_error_rate_counts_insertions_and_substitutions():
    assert word_error_rate(["TÔI MUỐN NƯỚC"], ["TÔI CẦN NƯỚC HÔM NAY"]) == 1.0
