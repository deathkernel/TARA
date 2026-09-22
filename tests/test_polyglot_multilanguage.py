from src.polyglot.generation import ModelCandidateGenerator
from src.polyglot.languages import default_languages


def test_generator_can_cycle_across_languages():
    prompts = []

    def fake_generate(prompt, **kwargs):
        prompts.append(prompt)
        language = prompt.split("Target language: ", 1)[1].split("\n", 1)[0]
        return f"```{language}\n// candidate\n```"

    languages = default_languages()[:3]
    generator = ModelCandidateGenerator(fake_generate, languages=languages, temperatures=(0.8,))
    candidates = tuple(generator("sorting", "", (), 5))

    assert [item.language for item in candidates] == ["python", "rust", "cpp"]
    assert len(prompts) == 5
