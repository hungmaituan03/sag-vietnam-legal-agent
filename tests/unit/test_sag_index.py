"""Unit tests for the SAG v0 event/entity index.

Chunks are built directly (not through the chunker) so these tests fail only
when the index changes. One integration test at the end feeds real chunker
output in, to catch label/key mismatches.
"""

import json
from pathlib import Path

import numpy as np

from sag_legal.ingestion import flatten_chunks, ingest_corpus
from sag_legal.models import LegalChunk
from sag_legal.sag import build_index, build_semantic_edges, entity_keys, expand


def _chunk(
    doc: str,
    article: str | None = None,
    clause: str | None = None,
    point: str | None = None,
    text: str | None = None,
) -> LegalChunk:
    """Mirror the chunker's id format: doc::Điều2::Khoản1::Điểma."""
    parts = [doc]
    for label in (article, clause, point):
        if label:
            parts.append(label.replace(" ", ""))
    if len(parts) == 1:
        parts.append("body")
    labels = " ".join(p for p in (article, clause, point) if p)
    return LegalChunk(
        chunk_id="::".join(parts),
        document_id=doc,
        text=text or labels or "nội dung",
        article=article,
        clause=clause,
        point=point,
    )


def _mini_chunks() -> list[LegalChunk]:
    """Two laws that both contain Điều 1/2 — the collision that proves scoping."""
    return [
        _chunk("doc-a", "Điều 1"),
        _chunk("doc-a", "Điều 2"),
        _chunk("doc-a", "Điều 2", "Khoản 1"),
        _chunk("doc-a", "Điều 2", "Khoản 1", "Điểm a"),
        _chunk("doc-a", "Điều 2", "Khoản 1", "Điểm b"),
        _chunk("doc-a", "Điều 2", "Khoản 2"),
        _chunk("doc-b", "Điều 1"),
        _chunk("doc-b", "Điều 2"),
        _chunk("doc-b", "Điều 2", "Khoản 1"),
    ]


# --- entity_keys ----------------------------------------------------------


def test_entity_keys_for_point_chunk_are_scoped_by_document():
    chunk = _chunk("doc-a", "Điều 2", "Khoản 1", "Điểm a")
    assert entity_keys(chunk) == [
        "law::doc-a",
        "art::doc-a::Điều 2",
        "cls::doc-a::Điều 2::Khoản 1",
    ]


def test_entity_keys_article_only_chunk_has_no_clause_key():
    assert entity_keys(_chunk("doc-a", "Điều 2")) == ["law::doc-a", "art::doc-a::Điều 2"]


def test_entity_keys_document_level_chunk_has_only_law_key():
    keys = entity_keys(_chunk("doc-a"))
    assert keys == ["law::doc-a"]


# --- build_index ----------------------------------------------------------


def test_build_index_maps_id_to_the_chunk_object():
    chunks = _mini_chunks()
    index = build_index(chunks)

    assert len(index.events_by_id) == len(chunks)
    for chunk in chunks:
        assert index.events_by_id[chunk.chunk_id] is chunk


def test_build_index_article_bucket_lists_every_descendant():
    index = build_index(_mini_chunks())

    assert index.events_by_entity["art::doc-a::Điều 2"] == [
        "doc-a::Điều2",
        "doc-a::Điều2::Khoản1",
        "doc-a::Điều2::Khoản1::Điểma",
        "doc-a::Điều2::Khoản1::Điểmb",
        "doc-a::Điều2::Khoản2",
    ]


def test_build_index_keys_do_not_merge_across_documents():
    index = build_index(_mini_chunks())

    assert index.events_by_entity["art::doc-a::Điều 1"] == ["doc-a::Điều1"]
    assert index.events_by_entity["art::doc-b::Điều 1"] == ["doc-b::Điều1"]
    assert len(index.events_by_entity["law::doc-b"]) == 3


def test_build_index_duplicate_chunk_id_keeps_first_and_does_not_double_bucket():
    first = _chunk("doc-a", "Điều 2", "Khoản 1", text="bản gốc")
    duplicate = _chunk("doc-a", "Điều 2", "Khoản 1", text="bản trùng")
    index = build_index([first, duplicate])

    assert len(index.events_by_id) == 1
    assert index.events_by_id[first.chunk_id].text == "bản gốc"
    assert index.events_by_entity["art::doc-a::Điều 2"] == [first.chunk_id]


# --- expand ---------------------------------------------------------------


def test_expand_adds_same_article_siblings_after_the_seed():
    index = build_index(_mini_chunks())
    seed = index.events_by_id["doc-a::Điều2::Khoản1::Điểma"]

    out = expand([seed], index)
    ids = [c.chunk_id for c in out]

    assert ids[0] == seed.chunk_id
    assert len(ids) == len(set(ids))
    assert set(ids) == set(index.events_by_entity["art::doc-a::Điều 2"])


def test_expand_two_seeds_in_one_article_add_no_duplicates():
    index = build_index(_mini_chunks())
    seeds = [
        index.events_by_id["doc-a::Điều2::Khoản1::Điểma"],
        index.events_by_id["doc-a::Điều2::Khoản2"],
    ]

    out = expand(seeds, index)
    ids = [c.chunk_id for c in out]

    assert len(ids) == len(set(ids))
    assert ids[:2] == [s.chunk_id for s in seeds]
    assert len(ids) == 5


def test_expand_never_crosses_documents():
    index = build_index(_mini_chunks())
    seed = index.events_by_id["doc-b::Điều2::Khoản1"]

    out = expand([seed], index)

    assert {c.document_id for c in out} == {"doc-b"}
    assert {c.chunk_id for c in out} == {"doc-b::Điều2::Khoản1", "doc-b::Điều2"}


def test_expand_respects_max_extra():
    index = build_index(_mini_chunks())
    seed = index.events_by_id["doc-a::Điều2::Khoản1::Điểma"]

    out = expand([seed], index, max_extra=1)

    assert len(out) == 2
    assert out[0] is seed


def test_expand_seed_without_article_returns_seeds_unchanged():
    index = build_index(_mini_chunks())
    orphan = _chunk("doc-a")

    assert expand([orphan], index) == [orphan]


def test_expand_empty_seeds_returns_empty():
    assert expand([], build_index(_mini_chunks())) == []


# --- semantic edges -------------------------------------------------------


def _cross_law_chunks() -> list[LegalChunk]:
    return [
        _chunk("doc-a", "Điều 1"),
        _chunk("doc-b", "Điều 1"),
        _chunk("doc-b", "Điều 2"),
    ]


def _cross_law_vectors() -> dict[str, np.ndarray]:
    """doc-a::Điều1 and doc-b::Điều1 are near-identical; doc-b::Điều2 is not."""
    return {
        "doc-a::Điều1": np.array([1.0, 0.0], dtype=np.float32),
        "doc-b::Điều1": np.array([0.98, 0.199], dtype=np.float32),
        "doc-b::Điều2": np.array([0.0, 1.0], dtype=np.float32),
    }


def test_build_semantic_edges_links_nearest_neighbour_across_documents():
    edges = build_semantic_edges(
        _cross_law_chunks(), _cross_law_vectors(), top_n=1, min_sim=0.5
    )

    assert [nid for nid, _ in edges["doc-a::Điều1"]] == ["doc-b::Điều1"]
    assert edges["doc-a::Điều1"][0][1] > 0.9
    assert "doc-b::Điều2" not in edges, "its best match is below min_sim"


def test_build_semantic_edges_never_links_a_chunk_to_itself():
    edges = build_semantic_edges(
        _cross_law_chunks(), _cross_law_vectors(), top_n=2, min_sim=-1.0
    )

    for chunk_id, neighbours in edges.items():
        assert chunk_id not in [nid for nid, _ in neighbours]


def test_build_semantic_edges_min_sim_can_drop_everything():
    edges = build_semantic_edges(
        _cross_law_chunks(), _cross_law_vectors(), top_n=1, min_sim=0.999
    )
    assert edges == {}


def test_build_semantic_edges_drops_near_duplicates():
    """A verbatim restatement adds nothing, so it must not hold a slot."""
    chunks = [
        _chunk("doc-a", "Điều 1"),
        _chunk("doc-b", "Điều 1"),  # identical vector to doc-a
        _chunk("doc-b", "Điều 2"),  # merely similar
    ]
    vectors = {
        "doc-a::Điều1": np.array([1.0, 0.0], dtype=np.float32),
        "doc-b::Điều1": np.array([1.0, 0.0], dtype=np.float32),
        "doc-b::Điều2": np.array([0.95, 0.312], dtype=np.float32),
    }

    edges = build_semantic_edges(chunks, vectors, top_n=1, min_sim=0.5, max_sim=0.99)
    neighbours = [nid for nid, _ in edges["doc-a::Điều1"]]

    assert "doc-b::Điều1" not in neighbours, "cosine 1.0 duplicate must be dropped"
    assert neighbours == ["doc-b::Điều2"], "the freed slot goes to a real neighbour"


def test_build_semantic_edges_keeps_close_but_distinct_neighbours():
    chunks = [_chunk("doc-a", "Điều 1"), _chunk("doc-b", "Điều 1")]
    vectors = {
        "doc-a::Điều1": np.array([1.0, 0.0], dtype=np.float32),
        "doc-b::Điều1": np.array([0.95, 0.312], dtype=np.float32),
    }

    edges = build_semantic_edges(chunks, vectors, top_n=1, min_sim=0.5, max_sim=0.99)

    assert [nid for nid, _ in edges["doc-a::Điều1"]] == ["doc-b::Điều1"]


def test_build_index_without_vectors_has_no_semantic_edges():
    assert build_index(_mini_chunks()).neighbours == {}


def test_build_index_attaches_semantic_edges_when_given_vectors():
    index = build_index(
        _cross_law_chunks(), vectors=_cross_law_vectors(), top_n=1, min_sim=0.5
    )
    assert "doc-b::Điều1" in [nid for nid, _ in index.neighbours["doc-a::Điều1"]]


# --- frontier expansion ---------------------------------------------------


def _chained_index():
    """doc-a::Điều1 -> doc-b::Điều1 -> doc-b::Điều2, one semantic edge each."""
    index = build_index(_cross_law_chunks())
    index.neighbours = {
        "doc-a::Điều1": [("doc-b::Điều1", 0.98)],
        "doc-b::Điều1": [("doc-b::Điều2", 0.71)],
    }
    return index


def test_expand_follows_a_semantic_edge_into_another_law():
    index = _chained_index()
    seed = index.events_by_id["doc-a::Điều1"]

    out = expand([seed], index)

    assert [c.chunk_id for c in out] == ["doc-a::Điều1", "doc-b::Điều1"]


def test_expand_can_be_restricted_to_structural_edges():
    index = _chained_index()
    seed = index.events_by_id["doc-a::Điều1"]

    assert expand([seed], index, use_semantic=False) == [seed]


def test_expand_two_hops_reaches_further_than_one():
    index = _chained_index()
    seed = index.events_by_id["doc-a::Điều1"]

    one_hop = [c.chunk_id for c in expand([seed], index, hops=1)]
    two_hops = [c.chunk_id for c in expand([seed], index, hops=2)]

    assert one_hop == ["doc-a::Điều1", "doc-b::Điều1"]
    assert two_hops == ["doc-a::Điều1", "doc-b::Điều1", "doc-b::Điều2"]


def test_expand_min_sim_filters_weak_edges_at_query_time():
    index = _chained_index()
    seed = index.events_by_id["doc-a::Điều1"]

    out = expand([seed], index, hops=2, min_sim=0.8)

    assert [c.chunk_id for c in out] == ["doc-a::Điều1", "doc-b::Điều1"]


def _deep_article_chunks() -> list[LegalChunk]:
    """One Điều where the seed's own khoản sits late in document order."""
    return [
        _chunk("doc-a", "Điều 2"),
        _chunk("doc-a", "Điều 2", "Khoản 1"),
        _chunk("doc-a", "Điều 2", "Khoản 1", "Điểm a"),
        _chunk("doc-a", "Điều 2", "Khoản 1", "Điểm b"),
        _chunk("doc-a", "Điều 2", "Khoản 2"),
        _chunk("doc-a", "Điều 2", "Khoản 2", "Điểm a"),
        _chunk("doc-a", "Điều 2", "Khoản 2", "Điểm b"),
    ]


def test_expand_reaches_the_seeds_own_clause_before_distant_siblings():
    index = build_index(_deep_article_chunks())
    seed = index.events_by_id["doc-a::Điều2::Khoản2::Điểma"]

    out = expand([seed], index, max_extra=2)

    assert [c.chunk_id for c in out] == [
        "doc-a::Điều2::Khoản2::Điểma",
        "doc-a::Điều2",
        "doc-a::Điều2::Khoản2",
    ], "heading and the seed's own khoản must come before Khoản 1's subtree"


def test_expand_keeps_document_order_within_a_kinship_rank():
    index = build_index(_deep_article_chunks())
    seed = index.events_by_id["doc-a::Điều2::Khoản2::Điểma"]

    tail = [c.chunk_id for c in expand([seed], index, max_extra=10)][3:]

    assert tail == [
        "doc-a::Điều2::Khoản2::Điểmb",
        "doc-a::Điều2::Khoản1",
        "doc-a::Điều2::Khoản1::Điểma",
        "doc-a::Điều2::Khoản1::Điểmb",
    ]


def test_expand_shares_a_tight_budget_across_seeds():
    """A seed in a big article must not starve the other seeds."""
    index = build_index(_mini_chunks())
    seeds = [
        index.events_by_id["doc-a::Điều2::Khoản1::Điểma"],  # 4 unseen siblings
        index.events_by_id["doc-b::Điều2::Khoản1"],  # 1 unseen sibling
    ]

    out = expand(seeds, index, max_extra=2)
    added = [c.document_id for c in out if c not in seeds]

    assert sorted(added) == ["doc-a", "doc-b"]


def test_expand_budget_still_caps_a_multi_hop_walk():
    index = _chained_index()
    seed = index.events_by_id["doc-a::Điều1"]

    assert len(expand([seed], index, hops=5, max_extra=1)) == 2


# --- integration with the real chunker ------------------------------------


def test_build_index_and_expand_on_real_chunker_output(tmp_path: Path):
    rows = [
        {
            "id": "mini-tctd",
            "title": "Luật mẫu TCTD",
            "type": "law",
            "content": (
                "Điều 1. Phạm vi\n"
                "Luật này quy định về tổ chức tín dụng.\n"
                "Điều 2. Giấy phép\n"
                "1. Điều kiện cấp giấy phép.\n"
                "a) Nhận tiền gửi.\n"
                "b) Cấp tín dụng.\n"
            ),
        },
    ]
    corpus = tmp_path / "mini.json"
    corpus.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")

    chunks = flatten_chunks(ingest_corpus(corpus, doc_ids=("mini-tctd",)))
    index = build_index(chunks)

    assert "art::mini-tctd::Điều 2" in index.events_by_entity

    seed = next(c for c in chunks if c.point == "Điểm a")
    paths = {c.citation_path for c in expand([seed], index)}

    assert "Điều 2, Khoản 1, Điểm a" in paths
    assert "Điều 2, Khoản 1, Điểm b" in paths
    assert "Điều 1" not in paths
