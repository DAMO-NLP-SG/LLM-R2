# @TIME : 15/6/23 4:05 PM
# @AUTHOR : LZDH

import psycopg2
import json
import time

# db_id = 'browser_web'
# new_db_name = db_id + 'test'
# gpt
# test_query = "SELECT browser.name FROM browser INNER JOIN (SELECT * FROM accelerator_compatible_browser WHERE compatible_since_year > 1998) AS t ON browser.id = t.browser_id INNER JOIN (SELECT * FROM Web_client_accelerator WHERE name = 'CProxy') AS t0 ON t.accelerator_id = t0.id;"
# orginal
# test_query = "SELECT T1.name FROM browser AS T1 JOIN accelerator_compatible_browser AS T2 ON T1.id  =  T2.browser_id JOIN " \
#         "web_client_accelerator AS T3 ON T2.accelerator_id  =  T3.id WHERE T3.name  =  'CProxy' AND " \
#         "T2.compatible_since_year  >  1998; "
# LR
# test_query = "SELECT browser10.name FROM browser AS browser10 INNER JOIN (SELECT * FROM accelerator_compatible_browser WHERE compatible_since_year > 1998) AS t38 ON browser10.id = t38.browser_id INNER JOIN (SELECT * FROM Web_client_accelerator WHERE name = 'CProxy') AS t39 ON t38.accelerator_id = t39.id;"


def query_cost_postgres(db_id, test_query):
    with open('data/schemas_100000/' + db_id + '.json', 'r') as f_sche:
        data = f_sche.read()
        table_details = json.loads(data)
    table_names = [i['table'] for i in table_details]
    # print(table_names)

    # Connect to the PostgreSQL database
    pg_conn = psycopg2.connect(
        host="localhost",
        port="5432",
        database=db_id,
        user="postgres",
        password="19981212lzdhMJK"
    )
    # disable_query_cache(pg_conn)
    pg_cursor = pg_conn.cursor()
    clr_q = "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE pid <> pg_backend_pid() AND datname = '" + db_id + "';"
    pg_cursor.execute(clr_q)
    pg_cursor.execute("SET enable_seqscan = off;")
    pg_cursor.execute("SET enable_indexscan = off;")
    try:
        error = ''
        # Commit the changes and close the connections
        start_time = time.time()
        print('in')
        pg_cursor.execute(test_query)
        pg_conn.commit()
        # results = pg_cursor.fetchall()
        end_time = time.time()
        execution_time = end_time - start_time
        print('stop')

        # print("Query Results:")
        # for row in results:
        #     print(row)
        # print("Execution Time:", execution_time)
        pg_cursor.close()
        pg_conn.close()
        return execution_time, error
    except Exception as error1:
        print('error')
        pg_cursor.close()
        pg_conn.close()
        execution_time = 'NA'
        return execution_time, error1




# query_cost_postgres(db_id, test_query)
