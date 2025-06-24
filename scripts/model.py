import unsloth

from unsloth.chat_templates import get_chat_template
from unsloth import FastLanguageModel


from constants import *

# Classe base para realizar Fine-Tuning ou Inferência do modelo.
class Model:
    
    def __init__(self, model_path, device="cuda"):
        self.model_path = model_path
        self.device = device

        self._initialize()
        self._initialize_tokenizer()
    
    def _initialize(self):

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            self.model_path,
            dtype = None,
            load_in_4bit = True,
            load_in_8bit = False, # [NEW!] A bit more accurate, uses 2x memory
            full_finetuning = False, # [NEW!] We have full finetuning now!
            device_map="auto",
            max_seq_length=8000,
        )

        # Configuração base para carregar o modelo utilizando Unsloth. Segundo a documentação esta é a forma otimizada de se carregar.
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",],
            r = 16,
            lora_alpha = 16,
            lora_dropout = 0.2,
            bias = "none",
            use_gradient_checkpointing = "unsloth", 
            random_state = 3047,
            use_rslora = False,
            loftq_config = None,
        ).to(self.device)

    def _initialize_tokenizer(self):
        
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id

        self.tokenizer = get_chat_template(
            self.tokenizer,
            chat_template = "llama-3",
        )

    # Aplica o chat template no prompt e tokeniza.
    def _apply_template(self, data):

        prompt = self.tokenizer.apply_chat_template(
            data,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )

        return prompt
