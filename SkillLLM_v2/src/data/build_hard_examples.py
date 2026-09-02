from __future__ import annotations

import argparse
import random
from dataclasses import dataclass, field

from src.data.dataset_utils import unique_skill_texts
from src.runtime_utils import read_jsonl, write_jsonl


@dataclass
class SyntheticExampleBuilder:
    example_id: str
    text_parts: list[str] = field(default_factory=list)
    skills: list[dict[str, object]] = field(default_factory=list)
    cursor: int = 0

    def add_text(self, value: str) -> None:
        self.text_parts.append(value)
        self.cursor += len(value)

    def add_skill(self, value: str, label: str) -> None:
        start = self.cursor
        self.text_parts.append(value)
        self.cursor += len(value)
        self.skills.append({"text": value, "label": label, "start": start, "end": self.cursor})

    def build(self, augmentation_type: str) -> dict[str, object]:
        return {
            "id": self.example_id,
            "text": "".join(self.text_parts),
            "skills": self.skills,
            "augmentation_type": augmentation_type,
        }


def sample_text(pool: list[str], rng: random.Random, minimum_words: int = 1) -> str:
    candidates = [item for item in pool if len(item.split()) >= minimum_words]
    if not candidates:
        candidates = pool
    return rng.choice(candidates)


def adjacent_example(example_id: str, hard_pool: list[str], soft_pool: list[str], rng: random.Random) -> dict[str, object]:
    builder = SyntheticExampleBuilder(example_id)
    builder.add_skill(sample_text(hard_pool, rng, minimum_words=1), "hard_skill")
    builder.add_text(", ")
    builder.add_skill(sample_text(soft_pool, rng, minimum_words=2), "soft_skill")
    builder.add_text(", and ")
    builder.add_skill(sample_text(hard_pool, rng, minimum_words=2), "hard_skill")
    builder.add_text(" are required for this role.")
    return builder.build("adjacent_mentions")


def repeated_example(example_id: str, soft_pool: list[str], hard_pool: list[str], rng: random.Random) -> dict[str, object]:
    repeated = sample_text(soft_pool, rng, minimum_words=1)
    builder = SyntheticExampleBuilder(example_id)
    builder.add_skill(repeated, "soft_skill")
    builder.add_text(" is important, and stronger ")
    builder.add_skill(repeated.lower(), "soft_skill")
    builder.add_text(" plus ")
    builder.add_skill(sample_text(hard_pool, rng, minimum_words=1), "hard_skill")
    builder.add_text(" experience will help.")
    return builder.build("repeated_mentions")


def long_sentence_example(example_id: str, soft_pool: list[str], hard_pool: list[str], rng: random.Random) -> dict[str, object]:
    soft_skill = sample_text(soft_pool, rng, minimum_words=2)
    hard_skill = sample_text(hard_pool, rng, minimum_words=2)
    builder = SyntheticExampleBuilder(example_id)
    builder.add_text("While coordinating a fast-moving backlog across product, engineering, and support, you must show ")
    builder.add_skill(soft_skill, "soft_skill")
    builder.add_text(" and use ")
    builder.add_skill(hard_skill, "hard_skill")
    builder.add_text(" without losing delivery quality.")
    return builder.build("soft_skill_in_long_sentence")


def near_miss_example(example_id: str, hard_pool: list[str], soft_pool: list[str], rng: random.Random) -> dict[str, object]:
    hard_skill = sample_text(hard_pool, rng, minimum_words=1)
    soft_skill = sample_text(soft_pool, rng, minimum_words=1)
    builder = SyntheticExampleBuilder(example_id)
    builder.add_text('General exposure to "')
    builder.add_text(f"{hard_skill.lower()}-related")
    builder.add_text('" tasks is not enough; hands-on ')
    builder.add_skill(hard_skill, "hard_skill")
    builder.add_text(" and ")
    builder.add_skill(soft_skill, "soft_skill")
    builder.add_text(" are expected.")
    return builder.build("near_miss_negative")


def split_merge_example(example_id: str, hard_pool: list[str], rng: random.Random) -> dict[str, object]:
    left = sample_text(hard_pool, rng, minimum_words=1)
    right = sample_text(hard_pool, rng, minimum_words=1)
    builder = SyntheticExampleBuilder(example_id)
    builder.add_text("Experience with ")
    builder.add_skill(left, "hard_skill")
    builder.add_text("/")
    builder.add_skill(right, "hard_skill")
    builder.add_text(" tooling is preferred.")
    return builder.build("merged_split_edge_case")


def build_augmented_examples(train_examples: list[dict[str, object]], seed: int) -> list[dict[str, object]]:
    rng = random.Random(seed)
    hard_pool = unique_skill_texts(train_examples, label="hard_skill")
    soft_pool = unique_skill_texts(train_examples, label="soft_skill")
    if not hard_pool or not soft_pool:
        raise RuntimeError("Expected both hard_skill and soft_skill spans in the training data.")

    augmented: list[dict[str, object]] = []
    builders = (
        adjacent_example,
        repeated_example,
        long_sentence_example,
        near_miss_example,
        split_merge_example,
    )
    for index in range(max(50, len(train_examples) // 8)):
        builder = builders[index % len(builders)]
        example_id = f"aug_{index:05d}"
        if builder is adjacent_example:
            augmented.append(builder(example_id, hard_pool, soft_pool, rng))
        elif builder is repeated_example:
            augmented.append(builder(example_id, soft_pool, hard_pool, rng))
        elif builder is long_sentence_example:
            augmented.append(builder(example_id, soft_pool, hard_pool, rng))
        elif builder is near_miss_example:
            augmented.append(builder(example_id, hard_pool, soft_pool, rng))
        else:
            augmented.append(builder(example_id, hard_pool, rng))
    return augmented


def main() -> int:
    parser = argparse.ArgumentParser(description="Create hard-case augmented training examples.")
    parser.add_argument("--input-file", default="data/v2/train.jsonl", help="Base v2 training file.")
    parser.add_argument(
        "--output-file",
        default="data/v2/train_hard_augmented.jsonl",
        help="Augmented training file containing base and synthetic examples.",
    )
    parser.add_argument("--seed", default=13, type=int, help="Random seed.")
    args = parser.parse_args()

    train_examples = read_jsonl(args.input_file)
    augmented_examples = build_augmented_examples(train_examples, seed=args.seed)
    combined = list(train_examples) + augmented_examples
    write_jsonl(args.output_file, combined)

    print(f"Wrote {len(combined)} training examples to {args.output_file}")
    print(f"Added {len(augmented_examples)} synthetic hard examples.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
