import pytest

from Demo import parse
from Research.QuSpike_JEPA.mutations import REFERENCE, DELETION


def test_fasta_whitespace_ok():
    seq = REFERENCE[:600] + "\n" + REFERENCE[600:]
    p = parse.parse_sequence(">hdr desc\n" + seq + "\n")
    assert p.mutations == [] and p.length == len(REFERENCE)


def test_illegal_chars_report_position():
    seq = REFERENCE[:10] + "*" + REFERENCE[11:]
    with pytest.raises(parse.InputError) as e:
        parse.parse_sequence(seq)
    assert any("位置 11" in d["item"] for d in e.value.details)


def test_empty_and_short():
    for bad in ("", "   \n", ">only header", "ACDE"):
        with pytest.raises(parse.InputError):
            parse.parse_input(bad)


def test_truncated_warns():
    p = parse.parse_sequence(REFERENCE[:1000])
    assert any("截斷" in w for w in p.warnings)


def test_x_counted():
    seq = REFERENCE[:100] + "X" + REFERENCE[101:]
    p = parse.parse_sequence(seq)
    assert p.n_unknown == 1


def test_mutation_list_formats():
    p = parse.parse_mutation_list(f"{REFERENCE[451]}452R, T478K del69-70 H69-")
    ps = dict(p.mutations)
    assert ps[452] != DELETION and ps[69] == DELETION and ps[70] == DELETION


def test_mutation_list_errors_itemised():
    with pytest.raises(parse.InputError) as e:
        parse.parse_mutation_list("L452R, A452R, foo, L9999R")
    items = {d["item"] for d in e.value.details}
    assert {"A452R", "foo", "L9999R"} <= items and "L452R" not in items


def test_auto_detect():
    assert parse.detect_kind("L452R, T478K") == "mutations"
    assert parse.detect_kind(REFERENCE) == "sequence"
    assert parse.detect_kind(">x\nMFV") == "sequence"


def test_roundtrip_text():
    seq = REFERENCE[:451] + "R" + REFERENCE[452:]
    a = parse.parse_sequence(seq)
    b = parse.parse_mutation_list(parse.mutations_to_text(a.mutations))
    assert a.mutations == b.mutations
