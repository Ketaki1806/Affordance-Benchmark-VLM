from transformers import AutoModelForCausalLM, AutoTokenizer


class CaptionGenerationModel:
    def __init__(self):
        self.model_name = "Qwen/Qwen2.5-7B-Instruct"
        self.model = None
        self.tokenizer = None

    def load_model(self) -> None:
        # load the model and tokenizer
        # AutoModelForCausalLM this is the model class from transformers
        # from pretrained model we are using Qwen2.5-7B-Instruct
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype="auto",
            device_map="auto",
        )
        # AutoTokenizer this is the tokenizer class from transformers
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
    
    # check if the model and tokenizer are loaded
    def is_loaded(self) -> bool:
        return self.model is not None and self.tokenizer is not None

    # generate caption using the model
    # prompt is the input text for the model
    # return the generated caption
    def generate_caption(self, prompt: str) -> str:
        if not self.is_loaded():
            raise RuntimeError("Model is not loaded. Call load_model() first.")

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        # **inputs, take inputs as dict from tokenizer
        #     max_new_tokens=100, # maximum number of tokens to generate 100 words, length of caption
        #     num_return_sequences=1, # return 1 caption (test run, change to 2 later), also check how to generate hard negatives
        #     return_dict_in_generate=True, # return the generated tokens as a dictionary
        #     temperature=0.7, # temperature for the model, 0.7 is a good default value
        #     top_p=0.9, # top_p for the model, 0.9 is a good default value
        #     top_k=50, # top_k for the model, 50 is a good default value
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=100,
            num_return_sequences=1,
            return_dict_in_generate=True,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
        )
        # decode the generated tokens into a string
        return self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
