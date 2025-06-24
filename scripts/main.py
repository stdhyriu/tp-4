from training_model import TrainingModel

from constants import MODEL

from prompts.train.user import USER_PROMPT
from prompts.train.system import SYSTEM_PROMPT


# Realiza o treinamento do Adaptador por meio de Unsloth.
def main():
    model = TrainingModel(
        model_path=MODEL,
        dataset_path="dataset/train",
        dataset_name="spider_data.csv",
        )
    
    model.train(SYSTEM_PROMPT, USER_PROMPT)


if __name__ == "__main__":
    main()