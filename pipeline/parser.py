"""Free-text → Answers extraction using the already-loaded LLM.

In-prompt few-shot examples replace a vector-DB RAG pipeline: for the small
number of book-description patterns, three examples give equivalent quality at
zero infrastructure cost.  If the example library grows beyond ~20 entries,
replace ``_FEW_SHOT`` with a tiny FAISS index over stored examples and retrieve
the two closest ones at call time (the embed/ directory is reserved for this).
"""
from __future__ import annotations

import json

# Canonical defaults used when the LLM omits a field.
ANSWER_DEFAULTS: dict[str, str] = {
    "character_type": "child",
    "pronoun": "she/her",
    "hair_color": "brown",
    "skin_tone": "light",
    "favourite_animal": "cat",
    "loved_one": "Mom",
    "theme": "friendship",
    "setting": "magical forest",
    "art_style": "cartoon children's book",
    "favourite_activities": "",
}

_FEW_SHOT = """\
Example 1 —
Input: "Make a book about a little girl named Luna with blonde hair and light skin \
who loves cooking and dancing. Include her mom. Theme: being brave. Setting: enchanted forest."
Output: {"child_name":"Luna","character_type":"child","pronoun":"she/her",\
"hair_color":"blonde","skin_tone":"light","favourite_animal":"rabbit","loved_one":"Mom",\
"theme":"being brave","setting":"enchanted forest","art_style":"cartoon children's book",\
"favourite_activities":"cooking, dancing"}

Example 2 —
Input: "Rex is a green baby dinosaur who loves cooking in the jungle. \
His dad comes along. Art: watercolor."
Output: {"child_name":"Rex","character_type":"dinosaur","pronoun":"he/him",\
"hair_color":"green","skin_tone":"green","favourite_animal":"parrot","loved_one":"Dad",\
"theme":"adventure","setting":"jungle","art_style":"watercolor children's book",\
"favourite_activities":"cooking"}

Example 3 —
Input: "Story about Zara, an alien with purple skin who enjoys painting and reading. \
She explores outer space. Her grandma is in the story."
Output: {"child_name":"Zara","character_type":"alien","pronoun":"she/her",\
"hair_color":"purple","skin_tone":"purple","favourite_animal":"space cat",\
"loved_one":"Grandma","theme":"curiosity","setting":"outer space",\
"art_style":"cartoon children's book","favourite_activities":"painting, reading"}
"""

_SCHEMA = (
    "child_name, character_type (child / any creature), pronoun (she/her | he/him | they/them), "
    "hair_color (or body colour for non-humans), skin_tone (or body colour), "
    "favourite_animal (companion), loved_one (family member or empty string), "
    "theme (moral / lesson), setting (world / overall place), art_style, "
    "favourite_activities (comma-separated list of activities)"
)


def extraction_messages(text: str) -> list[dict]:
    """Chat messages that instruct the LLM to parse *text* into book fields."""
    user = (
        f"Extract children's book details from the description below.\n"
        f"Return valid JSON with exactly these keys: {_SCHEMA}.\n\n"
        f"Few-shot examples:\n{_FEW_SHOT}\n"
        f"Description: {text}\n\n"
        f"Return only the JSON object. Use sensible defaults for omitted fields."
    )
    return [
        {
            "role": "system",
            "content": (
                "You extract structured children's book details from free-text descriptions. "
                "Output only valid JSON, nothing else."
            ),
        },
        {"role": "user", "content": user},
    ]


def extract_answers_dict(text: str, llm) -> dict:
    """Call *llm* to parse *text* → Answers-compatible dict.

    Returns the raw parsed dict; callers should merge with ``ANSWER_DEFAULTS``
    to fill any keys the LLM omitted.
    """
    messages = extraction_messages(text)
    resp = llm.create_chat_completion(
        messages=messages,
        response_format={"type": "json_object"},
        max_tokens=512,
        temperature=0.1,
    )
    content = resp["choices"][0]["message"]["content"]
    return json.loads(content)
