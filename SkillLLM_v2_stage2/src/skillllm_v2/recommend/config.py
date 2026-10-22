"""Interpretable, serialisable configuration of the course recommender (ranker + candidate pool).

Every knob is a plain number / string so a configuration can be logged, diffed and re-run.  ``LEGACY_CONFIG``
reproduces the behaviour of the first version of the recommender (soft prerequisite score, 15-candidate pool,
list ends when no candidate covers a gap).  ``DEFAULT_CONFIG`` is the configuration frozen by the dev-half
tuning loop (see reports/skillllm_v2/recommender_loop_log.md); it is what ``CourseRecommendationPipeline``
uses unless told otherwise.

None of these knobs reads gold labels or relevance labels: they act on the user profile, the job requirements,
the course catalogue fields and the knowledge graph only.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any


@dataclass(frozen=True)
class RecommenderConfig:
    # --- interpretable score weights (composite score = sum of weighted terms, clipped to [0, 1]) -----------------
    w_gap: float = 0.40          # weighted gap coverage (newly covered gap mass / total gap mass)
    w_semantic: float = 0.15     # lexical overlap of course outcomes with job title + gap skill names
    w_prereq: float = 0.20       # prerequisite compatibility (share of prerequisites already satisfied)
    w_level: float = 0.15        # level suitability (course level vs learner level)
    w_goal: float = 0.10         # alignment of the course title with the learner's target role / job title
    w_redundancy: float = 0.15   # penalty: share of the course's skills already covered by selected courses
    w_workload: float = 0.05     # penalty: hours above the learner's stated budget
    w_unlock: float = 0.0        # bonus for a bridge course: gap mass of the blocked courses it makes startable

    # --- candidate pool -----------------------------------------------------------------------------------------
    retrieval_pool: int = 15     # hybrid-retrieval candidates handed to the ranker (a pool larger than the number of
                                 # gap-covering courses means the list is always filled to top_k when feasible)

    # --- prerequisite handling ----------------------------------------------------------------------------------
    # "soft"          : prerequisite compatibility is only a score term (legacy)
    # "strict_drop"   : a course is eligible only if every prerequisite is known or taught by an earlier pick
    # "strict_bridge" : as strict_drop, and a blocked course can be unlocked by first recommending a bridge course
    #                   (a startable course teaching the missing prerequisite); the bridge is placed before it
    prereq_mode: str = "soft"
    compat_basis: str = "chain"  # strict modes: "chain" = prerequisites satisfied by learner OR earlier picks count;
                                 # "learner" = only the learner's own skills count (prefers courses startable now)
    bridge_discount: float = 1.0  # multiplier on a blocked course's score when it is only reachable via a bridge

    # --- list construction --------------------------------------------------------------------------------------
    fill_related: bool = False   # when gap-covering/bridge candidates run out, append courses whose text mentions an open
                                 # gap skill (flagged role="related_text"), best score first, until top_k
    related_match: str = "phrase"  # "phrase": gap skill name occurs in the course text; "token": or shares a content word with a taught skill
    related_discount: float = 0.8  # multiplier on the score of such text-only matches
    redundancy_mode: str = "fraction"  # "fraction": |re-covered gaps| / |course skills| ; "mmr": |re-covered gap mass|
    min_score: float = 0.0       # stop (or skip) when the best remaining score is not above this value

    # --- learner level ------------------------------------------------------------------------------------------
    infer_user_level: bool = False  # infer beginner/intermediate/advanced from the declared skills when not given
    level_method: str = "catalog_depth"  # "skill_count" | "catalog_depth" (mean highest catalogue level of the learner's skills)
    level_depth_junior: float = 1.6  # catalog_depth <= this -> beginner
    level_depth_senior: float = 2.1  # catalog_depth >= this -> advanced
    level_senior_skills: int = 7    # >= this many declared skills -> "advanced"
    level_junior_skills: int = 4    # <= this many declared skills -> "beginner"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def with_(self, **changes: Any) -> "RecommenderConfig":
        return replace(self, **changes)


LEGACY_CONFIG = RecommenderConfig()

# Frozen by the dev-half tuning loop (reports/skillllm_v2/recommender_loop_log.md, final config).  Rounds adopted:
#   R1 strict prerequisites + bridge sequencing, R2 related-text fill, R3 MMR redundancy w=0.10, R5 w_prereq=0.30.
#   Round 4 (learner-level inference) was rejected on dev, so infer_user_level stays False.
DEFAULT_CONFIG = LEGACY_CONFIG.with_(
    # R1
    prereq_mode="strict_bridge",
    compat_basis="learner",
    w_unlock=0.2,
    # R2
    fill_related=True,
    related_match="token",
    related_discount=0.8,
    retrieval_pool=40,
    # R3
    redundancy_mode="mmr",
    w_redundancy=0.10,
    # R5
    w_prereq=0.30,
)
