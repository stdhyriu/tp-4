import torch

from model import Model
from constants import *


class EvalMMLUModel(Model):
    def __init__(self, model_path, device="cuda"):
        super().__init__(model_path, device)
    
    # Loop pelo dataset para gerar o Prompt de cada dado de entrada
    def process_dataframe_dataset(self, data, system_prompt, user_prompt):

        # data.shuffle(seed=42)
        processed_data = []

        for _, row in data.iterrows():

            row["choices"] = "\n".join([str(count) + ". " + x  for count, x in enumerate(row["choices"])])
            
            input_data = {
                "subject": row["subject"],
                "question": row["question"],
                "choices": row["choices"],
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
            }

            aux_input = self.generate_prompt(**input_data)

            processed_data.append(aux_input)
                    
        # processed_data = Dataset.from_list(processed_data)

        return processed_data

    def _prompt(self, context_data, count):

        new_text = \
        f"""** Example {count} **
        Subject: {context_data["subject"]}
        Question: {context_data["question"]}
        Choices: {context_data["choices"]}
        Answer: {context_data["answer"]}\n"""

        return new_text

    def _system_prompt(self, data_examples, system_prompt):
        
        examples = """"""
        i = 1
        
        for _, row in data_examples.iterrows():
        
            row["choices"] = "\n".join([str(count) + ". " + x  for count, x in enumerate(row["choices"])])

            dict_question = {
                "subject": row["subject"],
                "question": row["question"],
                "choices": row["choices"],
                "answer": row["answer"],
            }

            examples = examples + self._prompt(dict_question, i) + "\n"
        
            i += 1
        
        return system_prompt + examples
        

    # Insere os dados gerados no Prompt
    def generate_prompt(self, subject, question, choices, system_prompt, user_prompt):

        input_prompt = f"""
        ### Instruction:
        {user_prompt}
        ### Input:
        Subject: {subject}
        Question: {question}
        Choices: {choices}
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

    def forward(self, data_input, data_examples, system_prompt, user_prompt, temperature=0.5, sample=True):

        __system_prompt = self._system_prompt(data_examples, system_prompt)
        list_prompt = self.process_dataframe_dataset(data_input, __system_prompt, user_prompt)

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
                max_new_tokens=10,
                use_cache=True,
            ).to(self.device)

        outputs = self.tokenizer.batch_decode(outputs, skip_special_tokes=True)

        return outputs