import inspect

import numpy as np
import pytest

from Demo import parse, predict as P
from Research.QuSpike_PQK import inference


def _x(bundle, s):
    return parse.to_vector(parse.parse_sequence(s), bundle.vocab)[0]


def test_matches_inference_predict(bundle, predictor, seqs):
    for s in seqs[:20]:
        pred = predictor.predict(s, k=22)
        x = _x(bundle, s)
        y, proba, _ = inference.predict(bundle.state, x, scale=pred.s_used)
        assert bundle.labels.index(pred.top_k[0].lineage) == int(y[0])
        for t in pred.top_k:
            assert t.prob == pytest.approx(proba[0, bundle.labels.index(t.lineage)], abs=1e-9)


def test_sorted_and_k(predictor, seqs):
    for k in (1, 3, 10, 22, 99):
        r = predictor.predict(seqs[0], k=k)
        assert len(r.top_k) == min(k, 22)
        ps = [t.prob for t in r.top_k]
        assert ps == sorted(ps, reverse=True) and sum(ps) <= 1 + 1e-9


def test_deterministic(predictor, seqs):
    assert predictor.predict(seqs[1]).to_dict() == predictor.predict(seqs[1]).to_dict()


def test_no_label_params():
    for fn in (P.Predictor.predict, P.Predictor._logits, parse.to_vector, parse.parse_input):
        names = set(inspect.signature(fn).parameters)
        assert not names & {"y", "label", "labels", "y_true", "y_test"}


def test_s_zero_no_residual(predictor, seqs):
    for s in seqs[:30]:
        r = predictor.predict(s, k=22, s=0)
        assert all(t.residual == 0 for t in r.top_k)
        assert all(t.prob == pytest.approx(t.ref_prob) for t in r.top_k)
        assert not r.top1_changed_by_expert and not r.routing.expert_applied


def test_residual_consistency_and_groups(predictor, seqs):
    seen = set()
    for s in seqs:
        r = predictor.predict(s, k=22, s=2.0)
        seen.add(r.routing.group)
        assert abs(sum(r.routing.probs.values()) - 1) < 1e-9
        for t in r.top_k:
            assert t.fused_logit - t.ref_logit == pytest.approx(t.residual)
            if r.routing.group == "O":
                assert t.residual == 0 and not r.routing.expert_applied
        if r.routing.group in ("D", "B"):
            assert r.routing.expert_applied
        changed = r.top_k[0].lineage != max(r.top_k, key=lambda t: t.ref_prob).lineage
        assert r.top1_changed_by_expert == changed
    assert seen <= {"D", "B", "O"} and "O" in seen


def test_nonzero_residual_in_quantum_group(predictor, seqs):
    nz = 0
    for s in seqs:
        r = predictor.predict(s, k=22, s=2.0)
        if r.routing.group in ("D", "B"):
            nz += any(abs(t.residual) > 1e-9 for t in r.top_k)
    assert nz > 0


def test_equivalent_mutation_list(bundle, predictor, seqs):
    for s in seqs[:20]:
        a = predictor.predict(s, "sequence", k=22)
        txt = parse.mutations_to_text(parse.parse_sequence(s).mutations) or "D614D"
        b = predictor.predict(txt, "mutations", k=22)
        assert [(t.lineage, t.prob) for t in a.top_k] == [(t.lineage, t.prob) for t in b.top_k]


def test_invalid_args(predictor, seqs):
    for kw in ({"k": 0}, {"s": -1.0}):
        with pytest.raises(parse.InputError):
            predictor.predict(seqs[0], **kw)
    with pytest.raises(parse.InputError):
        predictor.predict("")


def test_low_confidence_note(predictor):
    from Research.QuSpike_JEPA.mutations import REFERENCE
    r = predictor.predict(REFERENCE)
    assert r.confidence_note is None or "信心" in r.confidence_note or "詞彙" in r.confidence_note
