import os
import pandas as pd
import sqlite3

class CSVFolderToSQL:
    def __init__(self, folder_path, db_name):
        self.folder_path = folder_path
        self.db_name = db_name

    def csv_to_sql(self):
        conn = sqlite3.connect(self.db_name)
        for filename in os.listdir(self.folder_path):
            if filename.endswith('.csv'):
                file_path = os.path.join(self.folder_path, filename)
                table_name = os.path.splitext(filename)[0]  # Use the filename (without extension) as the table name
                
                df = pd.read_csv(file_path)
                df.to_sql(table_name, conn, if_exists='replace', index=False)
        conn.close()