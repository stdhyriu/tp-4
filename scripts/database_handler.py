import sqlite3


class DatabaseHandler(object):
    def __init__(self, database_name):

        self.database_name = database_name
        self._initialize()

    def _initialize(self):
        self.connection = sqlite3.connect(self.database_name)
        self.cursor = self.connection.cursor()
        
    def execute(self, sql):

        try:
            self.cursor.execute(sql)
            resp = self.cursor.fetchall()
        except:
            resp = None
        
        return resp

    def close_connection(self,):
        self.connection.close()



if __name__ == "__main__":

    for json_db in _f:
        name = json_db["db_id"]
        database_name = f"spider_data/database/{name}/{name}.sqlite"
        # sql = json_db["query"]
        
        print("### ###" * 15)
        database = DatabaseHandler(
            database_name=database_name,
            sql=json_db["query"]
        )

        database.db_commit()
        print()