import numpy as np
import pandas as pd

from tqdm import tqdm

from eval_mmlu_model import EvalMMLUModel
from constants import MODEL

from prompts.evaluation.user import USER_PROMPT
from prompts.evaluation.system import SYSTEM_PROMPT

from utils import list_dict_to_csv, process_output, process_output_2

# "experiments/checkpoint-1014"
# "experiments_2/checkpoint-2028"
# "experiments_3/checkpoint-1521"
# "experiments_4/checkpoint-1014"
# "experiments_5/checkpoint-1014"

def main(list_dataset_path, batch_size=10):

    eval_model = EvalMMLUModel(
            model_path="experiments_5/checkpoint-1014",
        )

    for dataset_path in list_dataset_path:
        
        list_results = []

        dataframe = pd.read_parquet(dataset_path)
        dataframe = dataframe.head(54)

        df_examples = dataframe.iloc[:4, :]
        __data = dataframe.iloc[4:, :]

        batch_data = np.array_split(__data, __data.shape[0]//batch_size)

        for data in tqdm(batch_data, total=len(batch_data)):

            list_output = []

            for _, row in data.iterrows():

                dict_output = {
                    "subject": row["subject"],
                    "question": row["question"],
                    "choices": row["choices"],
                    "answer": row["answer"],
                }

                list_output.append(dict_output)

            output = eval_model.forward(
                            data,
                            df_examples,
                            SYSTEM_PROMPT,
                            USER_PROMPT,
                            sample=False,
                        )

            for i, out in enumerate(output): 
                new_output = process_output_2(out)
                print("output ", new_output, list_output[i]["answer"])

                list_output[i]["model_output"] = new_output
                
            list_results = list_results + list_output
            context = list_results[0]["subject"]
        

        list_dict_to_csv(list_results, "results/eval/" + context + "/evaluation_results_experiment-5-1014.csv")



if __name__ == "__main__":
    
    list_dataset_path = [
            "dataset/mmlu/astronomy/test-astronomy.parquet",
            "dataset/mmlu/biology/test-biology.parquet",
            "dataset/mmlu/prehistory/test-prehistory.parquet"
        ]
    
    main(list_dataset_path)