import unsloth

from datasets import Dataset, load_dataset

from unsloth import is_bfloat16_supported
from transformers import TrainingArguments
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer

from model import Model
from constants import *


class TrainingModel(Model):
    def __init__(self, model_path, dataset_path, dataset_name, device="cuda"):
        super().__init__(model_path, device)
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
    
    # Loop pelo dataset para gerar o Prompt de cada dado de entrada
    def process_dataset(self, data, system_prompt, user_prompt):

        data.shuffle(seed=42)
        processed_data = []

        for row in data:
            
            input_data = {
                "question": row["question"],
                "query": row["query"],
                "database": row["db_id"],
                "tables": row["table_names"],
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
            }

            aux_dict = {
                "text": self.generate_prompt(**input_data)
            }

            processed_data.append(aux_dict)
                    
        processed_data = Dataset.from_list(processed_data)

        return processed_data

    # Insere os dados gerados no Prompt
    def generate_prompt(self, question, database, tables, query, system_prompt, user_prompt):

        input_prompt = f"""
        ### Instruction:
        {user_prompt}
        ### Input:
        Database: {database}
        Tables: {tables}
        Question: {question}
        ### Response:
        {query}
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
        list_prompt = data["text"]
        prompts = [self.tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=False) for prompt in list_prompt]

        return {"text" : prompts,}
    
    # Método para realizar o treinamento
    def train(self, system_prompt, user_prompt):

        dataset = load_dataset("csv", data_dir=self.dataset_path, data_files=self.dataset_name)
        dataset = dataset["train"].train_test_split(0.2)

        train_dataset = self.process_dataset(dataset["train"], system_prompt, user_prompt)
        train_dataset = train_dataset.map(self.formatting_prompts_function, batched = True)

        validation_dataset = self.process_dataset(dataset["test"], system_prompt, user_prompt)
        validation_dataset = validation_dataset.map(self.formatting_prompts_function, batched = True)

        response_template = "### Response:"
        response_template_ids = self.tokenizer.encode(response_template, add_special_tokens=False)[2:]
        collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=self.tokenizer)

        training_arguments = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            optim=optim,
            logging_strategy="epoch",
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            eval_strategy="epoch",
            save_strategy="epoch",
            max_grad_norm=max_grad_norm,
            max_steps=max_steps,
            warmup_ratio=warmup_ratio,
            group_by_length=group_by_length,
            lr_scheduler_type=lr_scheduler_type,
            report_to="tensorboard",
            seed=seed,
            warmup_steps=warmup_steps,
        )

        self.trainer = SFTTrainer(
            model=self.model,
            train_dataset=train_dataset,
            eval_dataset=validation_dataset,
            dataset_text_field="text",
            max_seq_length=max_seq_length,
            tokenizer=self.tokenizer,
            args=training_arguments,
            packing=packing,
            data_collator=collator,
        )

        self.trainer.train()