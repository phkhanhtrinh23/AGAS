from agas.cli import _rank_to_hr_ndcg


def test_rank_to_hr_ndcg_outside_k() -> None:
    hr, ndcg = _rank_to_hr_ndcg(rank=11, k=10)
    assert hr == 0.0
    assert ndcg == 0.0


def test_rank_to_hr_ndcg_rank1() -> None:
    hr, ndcg = _rank_to_hr_ndcg(rank=1, k=10)
    assert hr == 1.0
    assert ndcg == 1.0


def test_rank_to_hr_ndcg_rank2() -> None:
    hr, ndcg = _rank_to_hr_ndcg(rank=2, k=10)
    assert hr == 1.0
    assert 0.0 < ndcg < 1.0


def test_rank_to_hr_ndcg_invalid_rank() -> None:
    hr, ndcg = _rank_to_hr_ndcg(rank=0, k=10)
    assert hr == 0.0
    assert ndcg == 0.0
