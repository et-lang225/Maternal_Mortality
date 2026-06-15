import sqlite3
import pandas as pd

class SQLQuery:
    def __init__(self, db_name):
        self.db_name = db_name

    def execute_query(self, query):
        conn = sqlite3.connect(self.db_name)
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df