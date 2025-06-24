import csv
import json
import argparse

from database_handler import DatabaseHandler


def list_dict_to_csv(list_dict):

    keys = list_dict[0].keys()

    with open("spider_test_data.csv", "w", newline="", encoding="utf-8") as csv_file:
        dict_writer = csv.DictWriter(csv_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(list_dict)


def args():

    parser = argparse.ArgumentParser(description="Database Json processing")
    parser.add_argument('--database_path', type = str, help ='Choose a vailable Database path')
    parser.add_argument('--table_json', type = str, help ='Choose a vailable table json path')
    parser.add_argument('--spider_data', type = str, help ='Choose a vailable spider train data path')

    return parser.parse_args()


def main(database_path, table_json, spider_data):

    list_results = []
    dict_database = {}
    dict_table = {}

    with open(spider_data, "r") as js_file:
        query_json = json.load(js_file)
    
    with open(table_json, "r") as js_file:
        tables_json = json.load(js_file)
    
    for db_info in query_json:
        _exist = dict_database.get(db_info["db_id"], False)

        if _exist == False:
            _path = "/".join([database_path, db_info["db_id"], db_info["db_id"]+".sqlite"])
            dict_database[db_info["db_id"]] = DatabaseHandler(_path)

            for _table in tables_json: 
                if _table["db_id"] == db_info["db_id"]:
                    dict_table[db_info["db_id"]] = _table["table_names"]
                    break 
        
        _query = db_info["query"]
        response = dict_database[db_info["db_id"]].execute(_query)

        if response:
            # print(dict_table[db_info["db_id"]])
            list_results.append(
                {
                    "db_id": db_info["db_id"],
                    "query": _query,
                    "question": db_info["question"],
                    "table_names": ", ".join(dict_table[db_info["db_id"]]),
                    "response": response
                }
            )

    list_dict_to_csv(list_results)

    return

if __name__ == "__main__":

    # python generate_sql_outputs.py --database_path spider_data/database --table_json spider_data/tables.json --spider_data train_spider.json
    arguments = args()

    # print(arguments.database_path)
    
    database_path = arguments.database_path
    table_json = arguments.table_json
    spider_data = arguments.spider_data

    main(database_path, table_json, spider_data)
