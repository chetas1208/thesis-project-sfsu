from __future__ import annotations

from enum import Enum


class NodeType(str, Enum):
    USER = "User"
    PROFILE = "Profile"
    JOB = "Job"
    COURSE = "Course"
    SKILL = "Skill"
    SKILL_CLUSTER = "SkillCluster"
    OCCUPATION = "Occupation"
    PREREQUISITE = "Prerequisite"


class EdgeType(str, Enum):
    HAS_SKILL = "HAS_SKILL"
    MISSING_SKILL = "MISSING_SKILL"
    REQUIRES = "REQUIRES"
    PREFERS = "PREFERS"
    TEACHES = "TEACHES"
    REQUIRES_PREREQUISITE = "REQUIRES_PREREQUISITE"
    RELATED_TO = "RELATED_TO"
    SIMILAR_TO = "SIMILAR_TO"
    BELONGS_TO = "BELONGS_TO"
