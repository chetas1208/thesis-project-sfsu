from skillllm_v2.graph import SkillGraphBuilder
from skillllm_v2.graph.schema import EdgeType, NodeType


def test_graph_builder() -> None:
    builder = SkillGraphBuilder()
    builder.add_node("u1", NodeType.USER)
    builder.add_node("Python", NodeType.SKILL)
    builder.add_edge("u1", "Python", EdgeType.HAS_SKILL)
    assert builder.graph.number_of_edges() == 1
