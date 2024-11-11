import sqlite3

# Specify your database file path
db_path = "gflownet/tasks/logs/debug_run_seh_frag_2024-11-11_16-55-32/train/generated_objs_0.db"

# Connect to the database
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Execute a query to get the table names
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")

# Fetch and print all table names
tables = cursor.fetchall()
print("Tables in the database:", tables)

table_name = "results"
cursor.execute(f"PRAGMA table_info({table_name});")

# Fetch and print the schema
schema = cursor.fetchall()
print("Schema of the table:", schema)

cursor.execute(f"SELECT * FROM {table_name} LIMIT 5;")

# Fetch and print the rows
rows = cursor.fetchall()
for row in rows:
    print(row)
