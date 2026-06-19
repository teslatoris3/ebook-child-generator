"""Shared fixtures for the kids-ebook-generator test suite."""
import pytest
from pipeline.questionnaire import Answers


@pytest.fixture
def valid_answers() -> Answers:
    return Answers(
        child_name="Luna",
        pronoun="girl",
        hair_color="blonde",
        skin_tone="light",
        favourite_animal="dragon",
        loved_one="Mom",
        theme="being brave",
        setting="enchanted forest",
        art_style="watercolor children's book",
        favourite_activities="cooking, painting, bathing",
    )
