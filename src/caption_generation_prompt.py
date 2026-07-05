"""Build Qwen-VL prompts for affordance caption generation and negative regeneration."""

from pathlib import Path

from config_loader import PROJECT_ROOT, load_config


class CaptionGenerationPrompt:
    def __init__(self, prompt_template_path: str | None = None):
        config = load_config()
        default = PROJECT_ROOT / "configs" / "prompt_template.txt"
        self.prompt_template_path = Path(prompt_template_path or default)
        self.regen_template_path = PROJECT_ROOT / "configs" / "prompt_regeneration.txt"
        self.config = config

    def _read_template(self, path: Path) -> str:
        with open(path, encoding="utf-8") as f:
            return f.read().strip()

    def build_prompt(self, object_label: str) -> str:
        """Initial prompt: ask for most_probable + hard negative captions."""
        caps = self.config["captions"]
        template = self._read_template(self.prompt_template_path)
        return template.format(
            object_label=object_label,
            min_chars=caps["min_chars"],
            max_chars=caps["max_chars"],
            target_chars=caps["target_chars"],
            min_words=caps["min_words"],
            max_words=caps["max_words"],
            max_length_delta=caps["max_length_delta"],
            num_most_probable=caps["num_most_probable"],
            num_negative=caps["num_negative"],
        )

    def build_regeneration_prompt(
        self,
        object_label: str,
        rejected_negatives: list[str],
        positive_captions: list[str],
    ) -> str:
        """Follow-up prompt when CLIP filter rejects negatives as too easy."""
        caps = self.config["captions"]
        template = self._read_template(self.regen_template_path)
        rejected_list = "\n".join(f"- {c}" for c in rejected_negatives)
        positive_list = "\n".join(f"- {c}" for c in positive_captions)
        return template.format(
            object_label=object_label,
            rejected_list=rejected_list,
            positive_list=positive_list,
            num_negative=caps["num_negative"],
            min_chars=caps["min_chars"],
            max_chars=caps["max_chars"],
            min_words=caps["min_words"],
            max_words=caps["max_words"],
        )
