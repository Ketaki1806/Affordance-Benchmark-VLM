from pathlib import Path


class CaptionGenerationPrompt:
    def __init__(self, prompt_template_path: str):
        self.prompt_template_path = Path(prompt_template_path)

    # read prompt text from file
    def get_prompt(self) -> str:
        with open(self.prompt_template_path, "r", encoding="utf-8") as f:
            return f.read().strip()
