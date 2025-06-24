import numpy as np
import pandas as pd

from tqdm import tqdm

from constants import MODEL

from inference_model import InferenceModel
from prompts.few_shot.user import USER_PROMPT
from prompts.few_shot.system import SYSTEM_PROMPT

from utils import list_dict_to_csv, process_output


# Realiza o treinamento do Adaptador por meio de Unsloth.
def main(dataset_path, batch_size=10):

    list_results = []

    model = InferenceModel(
        # model_path=MODEL,
        model_path="experiments_5/checkpoint-1183"
        )

    dataframe = pd.read_csv(dataset_path)
    batch_data = np.array_split(dataframe, dataframe.shape[0]//batch_size)

    for data in tqdm(batch_data, total=len(batch_data)):

        list_output = []

        for _, row in data.iterrows():

            dict_output = {
              "db_id": row["db_id"], 
              "query": row["query"],
              "question": row["question"],
              "table_names": row["table_names"],
              "response": row["response"],
            }

            list_output.append(dict_output)

        output = model.forward(data, SYSTEM_PROMPT, USER_PROMPT)
        
        for i, out in enumerate(output): 
            new_output = process_output(out)
            list_output[i]["model_output"] = new_output
            
        list_results = list_results + list_output


    list_dict_to_csv(list_results, "results/fine_tuning_5_results--checkpoint--1183.csv")
    

if __name__ == "__main__":

    dataset_path = "dataset/test/spider_test_data.csv"
    
    main(dataset_path)