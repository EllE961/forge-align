from align import reward


def test_keyword_score_hits_and_misses():
    assert reward.keyword_score("a small puppy ran", "puppy") == 1.0
    assert reward.keyword_score("puppies played all day", "puppy") == 1.0
    assert reward.keyword_score("a small dog ran", "puppy") == 0.0


def test_ending_score_uses_only_the_tail():
    sad_then_happy = "She cried and cried. " * 3 + "Then she smiled and felt so happy."
    assert reward.ending_score(sad_then_happy, "happy") > 0.5
    assert reward.ending_score(sad_then_happy, "sad") < 0.5


def test_ending_score_zero_with_no_tone_words():
    assert reward.ending_score("The cat sat on the mat.", "happy") == 0.0


def test_length_score_peaks_at_target():
    exact = " ".join(["word"] * 60)
    assert reward.length_score(exact, target=60) == 1.0
    far = " ".join(["word"] * 10)
    assert reward.length_score(far, target=60) < reward.length_score(exact, target=60)


def test_diversity_score_penalises_loops():
    loop = "and then and then and then and then and then"
    varied = "the dragon flew over the misty green mountains at dawn"
    assert reward.diversity_score(loop) < reward.diversity_score(varied)


def test_score_combines_components_in_zero_one():
    rb = reward.score("The puppy played in the garden and felt very happy at the end.", "puppy", "happy")
    assert 0.0 <= rb.total <= 1.0
    assert rb.keyword == 1.0
