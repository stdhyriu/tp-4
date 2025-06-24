import re
import csv


def list_dict_to_csv(list_dict, destiny_path):

    keys = list_dict[0].keys()

    with open(destiny_path, "w", newline="", encoding="utf-8") as csv_file:
        dict_writer = csv.DictWriter(csv_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(list_dict)


def process_output(output):

    new_output = output.split("assistant<|end_header_id|>")[-1]
    new_output = new_output.split("<|eot_id|><|end_of_text|>")[0]
    new_output = re.sub("<\|finetune_right_pad_id\|>", "", new_output)

    return new_output.lstrip()


def process_output_2(output):

    new_output = output.split("### Response:")[-1]
    new_output = re.sub("<\|(.*)\|>", "", new_output)

    return new_output.lstrip()