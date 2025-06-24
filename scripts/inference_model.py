import torch

from datasets import Dataset, load_dataset

from model import Model
from constants import *


class InferenceModel(Model):
    def __init__(self, model_path, device="cuda"):
        super().__init__(model_path, device)
    
    # Loop pelo dataset para gerar o Prompt de cada dado de entrada
    def process_dataframe_dataset(self, data, system_prompt, user_prompt):

        processed_data = []

        for _, row in data.iterrows():
            
            input_data = {
                "question": row["question"],
                "database": row["db_id"],
                "tables": row["table_names"],
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
            }

            aux_input = self.generate_prompt(**input_data)

            processed_data.append(aux_input)
                    
        return processed_data

    # Insere os dados gerados no Prompt
    def generate_prompt(self, question, database, tables, system_prompt, user_prompt):

        input_prompt = f"""
        ### Instruction:
        {user_prompt}
        ### Input:
        Database: {database}
        Tables: {tables}
        Question: {question}
        ### Response:
        """

        prompt = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": input_prompt
            }
        ]

        return prompt

    # Aplica o Formato Template da LLM em todo o dataset
    def formatting_prompts_function(self, data):
        # list_prompt = data["text"]
        prompts = [self.tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=False) for prompt in data]

        return prompts

    def forward(self, data, system_prompt, user_prompt, temperature=0.5, sample=True):

        list_prompt = self.process_dataframe_dataset(data, system_prompt, user_prompt)

        # inputs = self._apply_template(new_prompt,).to(self.device)
        # inputs = self.formatting_prompts_function(new_prompt)
        inputs = self.tokenizer.apply_chat_template(
            list_prompt,
            tokenize=True,
            padding=True, 
            truncation=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(self.device)


        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                do_sample=sample,
                temperature=temperature,
                max_new_tokens=2048,
                use_cache=True,
            ).to(self.device)

        # list_response = []

        # for _out in outputs:
        #     response = self.tokenizer.decode(_out, skip_special_tokens=True)

        #     print(response)
        #     list_response.append(response)
        #     break

        outputs = self.tokenizer.batch_decode(outputs, skip_special_tokes=True)

        return outputs