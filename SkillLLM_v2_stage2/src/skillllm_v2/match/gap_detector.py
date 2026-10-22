from __future__ import annotations

from typing import Any

from skillllm_v2.match.skill_keys import SkillKeyResolver, skill_name
from skillllm_v2.recommend.gap_vector import GapEngine, GapVector
from skillllm_v2.schema import normalize_text


class GapDetector:
    """Legacy typed gap list (``detect``) and weighted gap vector (``detect_vector``).

    Skills are compared through :class:`skillllm_v2.match.skill_keys.SkillKeyResolver` (case/whitespace/punctuation
    insensitive, symbol-preserving, alias-aware), so ``python == Python`` and ``SQL == structured query language``.

    ``detect`` returns one entry per distinct job skill (a skill listed both as required and preferred is reported once,
    as required). Entries keep the legacy ``{"skill", "gap_type"[, "transferable_from"]}`` shape and add
    ``requirement_strength`` (``"required"`` / ``"preferred"``), ``skill_key`` and, for a skill also listed as preferred,
    ``also_listed_as_preferred``. Held skills that are weak / outdated / uncertain are reported for preferred skills too.
    ``covered_by_transferable_skill`` is only ever assigned to a skill the user does NOT hold.
    """

    def __init__(
        self,
        alias_resolver: Any | None = None,
        *,
        aliases: dict[str, str] | None = None,
        use_default_aliases: bool = True,
        key_resolver: SkillKeyResolver | None = None,
    ) -> None:
        self.engine = GapEngine()
        self.keys = key_resolver or SkillKeyResolver(alias_resolver, aliases=aliases, use_default_aliases=use_default_aliases)

    # ------------------------------------------------------------------ legacy typed gaps
    def detect(self, user_profile: dict[str, Any], job: dict[str, Any]) -> list[dict[str, Any]]:
        kr = self.keys
        held = kr.index(user_profile.get("skills"), user_profile.get("skill_entries"))
        weak = kr.index(user_profile.get("weak_skills"))
        outdated = kr.index(user_profile.get("outdated_skills"))
        uncertain = kr.index(user_profile.get("uncertain_skills"))
        transferable: dict[str, Any] = {}
        for source, origin in (user_profile.get("transferable_skills") or {}).items():
            for key in kr.keys(source):
                transferable.setdefault(key, origin)

        gaps: list[dict[str, Any]] = []
        by_key: dict[str, dict[str, Any]] = {}
        for strength, skills in (("required", job.get("required_skills")), ("preferred", job.get("preferred_skills"))):
            for item in _items(skills):
                keys = kr.keys(item)
                primary = kr.primary(item) or skill_name(item)
                if not keys:
                    continue
                earlier = by_key.get(primary) or next((by_key[k] for k in keys if k in by_key), None)
                if earlier is not None:
                    if strength == "preferred" and earlier["requirement_strength"] == "required":
                        earlier["also_listed_as_preferred"] = True
                    continue
                gap = self._classify(skill_name(item), primary, keys, strength, held, weak, outdated, uncertain, transferable)
                by_key[primary] = gap if gap is not None else {"requirement_strength": strength}
                for key in keys:
                    by_key.setdefault(key, by_key[primary])
                if gap is not None:
                    gaps.append(gap)
        return gaps

    @staticmethod
    def _classify(
        name: str,
        primary: str,
        keys: frozenset[str],
        strength: str,
        held: frozenset[str],
        weak: frozenset[str],
        outdated: frozenset[str],
        uncertain: frozenset[str],
        transferable: dict[str, Any],
    ) -> dict[str, Any] | None:
        base = {"skill": name, "requirement_strength": strength, "skill_key": primary}
        if not keys & held:
            origin = next((transferable[k] for k in sorted(keys) if k in transferable), None)
            if origin is not None:
                return {**base, "gap_type": "covered_by_transferable_skill", "transferable_from": origin}
            return {**base, "gap_type": "missing_critical" if strength == "required" else "missing_preferred"}
        if keys & weak:
            return {**base, "gap_type": "weak_skill"}
        if keys & outdated:
            return {**base, "gap_type": "outdated_skill"}
        if keys & uncertain:
            return {**base, "gap_type": "uncertain_skill"}
        return None

    # ------------------------------------------------------------------ weighted gap vector
    def detect_vector(self, user_profile: dict[str, Any], job: dict[str, Any], graph: Any = None) -> GapVector:
        """``GapEngine.compute`` on a profile whose held skills are re-keyed onto the job's skill ids.

        The engine matches on exact ``canonical_skill_id`` / ``normalize_text(name)``; the job's ids are kept as they were
        (so course linkage by skill id is unchanged) and every user skill that is the same skill under
        :class:`SkillKeyResolver` is presented to the engine under that id.
        """
        return self.engine.compute(self._align_profile(user_profile, job), job, graph=graph)

    def _requirement_ids(self, job: dict[str, Any]) -> list[tuple[str, frozenset[str]]]:
        out: list[tuple[str, frozenset[str]]] = []
        for field in ("required_skills", "preferred_skills"):
            for item in _items(job.get(field)):
                if isinstance(item, dict):
                    sid = item.get("canonical_skill_id") or normalize_text(str(item.get("skill_name") or item.get("name") or ""))
                else:
                    sid = normalize_text(str(item))
                if sid:
                    out.append((sid, self.keys.keys(item)))
        return out

    def _align_profile(self, user_profile: dict[str, Any], job: dict[str, Any]) -> dict[str, Any]:
        requirements = self._requirement_ids(job)
        if not requirements:
            return user_profile
        kr = self.keys

        def sids_for(item: Any) -> list[str]:
            keys = kr.keys(item)
            return list(dict.fromkeys(sid for sid, rkeys in requirements if keys & rkeys))

        mastery = self.engine.MASTERY_WEIGHTS
        best: dict[str, tuple[float, Any]] = {}
        passthrough: list[Any] = []
        tagged = [(item, True) for item in _items(user_profile.get("skills"))] + [(item, False) for item in _items(user_profile.get("skill_entries"))]
        for item, keep_if_unmatched in tagged:
            sids = sids_for(item)
            if not sids:
                if keep_if_unmatched:
                    passthrough.append(item)
                continue
            if isinstance(item, dict):
                try:
                    strength = float(item.get("confidence", 1.0)) * mastery.get(str(item.get("mastery_level") or item.get("mastery") or "proficient").lower(), 0.7)
                except (TypeError, ValueError):
                    strength = 0.0
                payload = dict(item)
            else:
                strength = 1.0
                payload = {"name": str(item), "confidence": 1.0, "mastery_level": "proficient"}
            for sid in sids:
                if sid not in best or strength > best[sid][0]:
                    best[sid] = (strength, {**payload, "canonical_skill_id": sid})
        aligned = dict(user_profile)
        aligned["skills"] = passthrough + [entry for _, entry in best.values()]
        aligned.pop("skill_entries", None)

        def remap(names: Any) -> list[Any]:
            out: list[Any] = []
            for name in _items(names):
                sids = sids_for(name)
                out.extend(sids if sids else [name])
            return out

        for field in ("weak_skills", "outdated_skills", "uncertain_skills"):
            if user_profile.get(field):
                aligned[field] = remap(user_profile[field])
        if user_profile.get("transferable_skills"):
            mapped: dict[Any, Any] = {}
            for source, origin in user_profile["transferable_skills"].items():
                for sid in sids_for(source) or [source]:
                    mapped.setdefault(sid, origin)
            aligned["transferable_skills"] = mapped
        return aligned


def _items(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (str, dict)):
        return [value]
    return list(value)
