from pathlib import Path

from caption_generation_prompt import CaptionGenerationPrompt
from logger import get_logger
from model import CaptionGenerationModel

PROMPT_TEMPLATE_PATH = (
    Path(__file__).resolve().parent.parent / "configs" / "prompt_template.txt"
)


class QWENCaptionGenerationPipeline:
    def __init__(self):
        self.logger = get_logger(__name__)
        self.prompt_template_path = str(PROMPT_TEMPLATE_PATH)
        self.model: CaptionGenerationModel | None = None
        self.prompt: str | None = None
        self.caption: str | None = None

    def load(self) -> None:
        self.logger.info("Loading caption generation prompt from %s", self.prompt_template_path)
        prompt_loader = CaptionGenerationPrompt(self.prompt_template_path)
        self.prompt = prompt_loader.get_prompt()

        self.logger.info("Loading QWEN model")
        self.model = CaptionGenerationModel()
        self.model.load_model()

    def validate_model(self) -> bool:
        if self.model is None or not self.model.is_loaded():
            self.logger.error("Model validation failed: model or tokenizer not loaded")
            return False

        self.logger.info("Model validation passed")
        return True

    def generate_caption(self) -> str:
        if self.model is None or self.prompt is None:
            raise RuntimeError("Pipeline not loaded. Call load() first.")

        self.logger.info("Generating caption")
        self.caption = self.model.generate_caption(self.prompt)
        return self.caption

    def validate_caption(self) -> bool:
        if self.caption is None or not self.caption.strip():
            self.logger.error("Caption validation failed: caption is empty")
            return False

        self.logger.info("Caption validation passed")
        return True

    def run(self) -> str:
        self.load()

        if not self.validate_model():
            raise RuntimeError("Model validation failed")

        caption = self.generate_caption()

        if not self.validate_caption():
            raise RuntimeError("Caption validation failed")

        return caption


if __name__ == "__main__":
    pipeline = QWENCaptionGenerationPipeline()
    print(pipeline.run())
