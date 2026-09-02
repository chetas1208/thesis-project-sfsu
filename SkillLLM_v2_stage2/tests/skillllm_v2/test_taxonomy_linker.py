from skillllm_v2.normalize import TaxonomyLinker


def test_taxonomy_linker_alias() -> None:
    match = TaxonomyLinker().link("AWS")
    assert match and match["canonical_skill_name"] == "Amazon Web Services"
