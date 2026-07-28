"""Randomized text-prompt paraphrases for the RJ45-cable-insertion dataset.

The dataset episodes are GRASPED-INSERTION-ONLY: the cable/connector is already
held in the gripper at frame 0, so there is no pick-up. The single fixed prompt
in datagen_to_lerobot.py ("pick up the ethernet cable and plug it into the jack")
is therefore misleading. This module generates natural paraphrases that describe
inserting an already-held/grasped cable into the jack.

All results are lowercase, grammatical, and have no trailing period, matching the
existing dataset style. Dependency-free (stdlib random / itertools only).

CLI:
  python tools/prompt_variations.py --n 20            # 20 random samples
  python tools/prompt_variations.py --all             # full deduped set + count
  python tools/prompt_variations.py --n 20 --seed 0   # reproducible samples
"""
import argparse
import itertools
import random

# --- combinatorial slots -----------------------------------------------------
# Structure: "<verb> <object phrase> <preposition + target>"

VERBS = ["insert", "plug", "put", "push", "seat", "connect"]

# base object nouns (the thing being inserted)
OBJECTS = [
    "cable",
    "ethernet cable",
    "rj45 cable",
    "rj45 connector",
    "connector",
    "plug",
]

# how the already-held object phrase is formed from a base noun.
# each entry is a function noun -> full noun phrase (kept grammatical + lowercase).
OBJECT_FORMS = [
    lambda n: f"the {n}",
    lambda n: f"the held {n}",
    lambda n: f"the grasped {n}",
    lambda n: f"the {n} you're holding",
]

# preposition + target
TARGETS = [
    "into the jack",
    "into the socket",
    "into the port",
    "in the jack",
    "inside the jack",
]

# fully hand-written natural variants (already grammatical, held-cable framing)
HANDWRITTEN = [
    "insert the held cable inside the jack",
    "insert the grasped cable into the jack",
    "plug the held ethernet cable into the port",
    "plug the cable you're holding into the jack",
    "seat the connector you're holding in the socket",
    "push the held rj45 plug into the jack",
    "guide the held cable into the jack",
    "guide the grasped connector into the socket",
    "line up the held plug and push it into the jack",
    "align the held connector with the jack and insert it",
    "fit the held rj45 connector into the port",
    "slot the held cable into the jack",
    "mate the held connector with the jack",
    "seat the held ethernet plug fully into the jack",
    "push the connector you're holding home into the jack",
]


def _combinatorial():
    """Yield every grammatical verb x object-form x object x target combination."""
    for verb, form, obj, target in itertools.product(
        VERBS, OBJECT_FORMS, OBJECTS, TARGETS
    ):
        yield f"{verb} {form(obj)} {target}"


def all_prompts():
    """Return the full deduplicated, deterministically-ordered prompt set."""
    seen = set()
    ordered = []
    for p in itertools.chain(_combinatorial(), HANDWRITTEN):
        if p not in seen:
            seen.add(p)
            ordered.append(p)
    ordered.sort()
    return ordered


_ALL = all_prompts()


def sample_prompt(rng=None):
    """Return one random prompt. Deterministic if a random.Random is passed."""
    r = rng if rng is not None else random
    return r.choice(_ALL)


def _main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=20, help="print N random samples")
    ap.add_argument("--all", action="store_true", help="print the full deduped set + count")
    ap.add_argument("--seed", type=int, default=None, help="rng seed for reproducibility")
    args = ap.parse_args()

    if args.all:
        prompts = all_prompts()
        for p in prompts:
            print(p)
        print(f"# {len(prompts)} distinct prompts")
        return

    rng = random.Random(args.seed)
    for _ in range(args.n):
        print(sample_prompt(rng))


if __name__ == "__main__":
    _main()


# -----------------------------------------------------------------------------
# To wire this into datagen_to_lerobot.py (do NOT edit that file here), the exact
# 2-line change is:
#
#   1) add an import near the top of datagen_to_lerobot.py:
#          from prompt_variations import sample_prompt
#      (or: from tools.prompt_variations import sample_prompt, depending on how
#       the converter is launched)
#
#   2) in the per-episode loop, seed a stable rng per episode and replace the
#      constant task. Concretely, at the top of `for ep_dir in complete:` add
#          rng = random.Random(hash(os.path.basename(ep_dir)) & 0xFFFFFFFF)
#      then change the add_frame call's
#          "task": PROMPT,
#      to
#          "task": sample_prompt(rng),
#      so a given episode keeps one stable prompt across all its frames while
#      different episodes get different prompts. (`import random` is already
#      needed; the stdlib is available in the converter's venv.)
# -----------------------------------------------------------------------------
