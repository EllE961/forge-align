from align.prompts import KEYWORDS, TONES, all_tasks, split_tasks


def test_all_tasks_cover_every_keyword_and_tone_combination():
    tasks = all_tasks()
    assert len(tasks) == len(KEYWORDS) * len(TONES)
    assert {(t.keyword, t.tone) for t in tasks} == {(k, tn) for k in KEYWORDS for tn in TONES}


def test_split_is_disjoint_and_reproducible():
    train1, eval1 = split_tasks(seed=0)
    train2, eval2 = split_tasks(seed=0)
    assert [t.prompt for t in train1] == [t.prompt for t in train2]
    train_set = {(t.keyword, t.tone) for t in train1}
    eval_set = {(t.keyword, t.tone) for t in eval1}
    assert train_set.isdisjoint(eval_set)
    assert len(train1) + len(eval1) == len(all_tasks())


def test_different_seed_gives_a_different_split():
    train_a, _ = split_tasks(seed=0)
    train_b, _ = split_tasks(seed=1)
    assert [t.prompt for t in train_a] != [t.prompt for t in train_b]
