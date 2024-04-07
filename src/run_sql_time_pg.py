# @TIME : 24/7/23 9:27 PM
# @AUTHOR : LZDH

import pandas as pd
import numpy as np
from run_postgre import query_cost_postgres
import psycopg2
from get_query_meta import *
import multiprocessing


def my_function(db_id, query, queue):
    result = query_cost_postgres(db_id, query)
    queue.put(result)


def get_execute_sql_detail(predicted_sql, ground_truth, db_id):
    if db_id == 'job_syn':
        db_id = 'job'
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
        pg_cursor.execute(predicted_sql)
        predicted_res = pg_cursor.fetchall()
        pg_cursor.execute(ground_truth)
        ground_truth_res = pg_cursor.fetchall()
        # print(ground_truth_res)
        time_ratio_pg = 0
        pg_cursor.close()
        pg_conn.close()
        if set(predicted_res) == set(ground_truth_res):
            return True
        else:
            return False
    except Exception as error1:
        print('error')
        pg_cursor.close()
        pg_conn.close()
        return False


# df_id = 'job_syn'
# df = pd.read_csv('gpt_job_syn_one_promo_random_withempty_t0.csv').fillna('NA')
# df_id = 'dsb'
# exps = ['plan']
# for exp in exps:
#     print(exp)
# 'tpch_50g 180',
df_ids = ['tpch_5g']
# df_ids = ['job_syn']
# exp = 'random'
# jk = 0
for df_id in df_ids:
    print(df_id)
    df = pd.read_csv('gpt_' + df_id + '_one_promo_queryCL_updated.csv').fillna('NA')
    df_b = pd.read_csv('results_gpt_tpch_5g_1shot_updated.csv').fillna('NA')
    df_t = pd.read_csv('results_baseline_tpch_1g_1shot_updated.csv').fillna('NA')
    # df_id = 'tpch'
    # df = pd.read_csv('gpt_' + df_id + '_one_promo_' + exp + '_withempty_t0.csv').fillna('NA')
    # df = pd.read_csv('gpt_dsb_one_promo_plan_updated.csv').fillna('NA')
    run_org = False

    if __name__ == '__main__':
        results = {}
        if df_id == 'tpch' or df_id == 'job_syn' or df_id == 'dsb' or df_id == 'tpch_1g' or df_id == 'tpch_5g':
            db_ids = df['db_id'].tolist()
            org_queries = df['original_sql'].tolist()

            gpt_queries = df['rewritten_sql_gpt'].tolist()
            # gpt_queries = df['rewritten_sql_LR'].tolist()
            # gpt_queries = [x.replace('t.*', '*') for x in gpt_queries]
            # gpt_rules = df['activated_rules_gpt'].tolist()
            # gpt_rules = df['activated_rules_LR'].tolist()
            # LR_queries = df['rewritten_sql_LR'].tolist()
            # LR_rules = df['activated_rules_LR'].tolist()
            latency_org = df_b['latency_org'].tolist()
            latency_test = df_t['latency_gpt'].tolist()
            # latency_org = []
            latency_gpt = []
            # latency_LR = []
            ids = []
            for i in range(len(db_ids)):
                if i >= 0:
                    print(i)
                    # print(gpt_queries[i])
                    ids.append(db_ids[i])
                    if df_id == 'job_syn':
                        fail_list = [25, 37, 71, 97, 126, 329, 333, 360, 366, 377, 392, 401, 406, 412, 440, 486]
                    else:
                        fail_list = []
                    if i in fail_list:
                        latency_org.append('NA')
                        latency_gpt.append('NA')
                        # latency_LR.append('NA')
                    else:
                        logical_plan_o = get_logical_plan(db_ids[i], org_queries[i])
                        logical_plan_g = get_logical_plan(db_ids[i], gpt_queries[i])
                        # print(logical_plan_o)
                        # print(logical_plan_g)
                        df_i = {}
                        l_org = []
                        l_gpt = []
                        # if run_org:
                        #     for j in range(5):
                        #         queue = multiprocessing.Queue()
                        #         p = multiprocessing.Process(target=my_function, args=(db_ids[i], org_queries[i], queue))
                        #         p.start()
                        #         # Set a timeout value
                        #         timeout = 300
                        #         # Wait for the function to finish or timeout
                        #         p.join(timeout)
                        #         # Check if the process is still running
                        #         if p.is_alive():
                        #             print("Function timed out. Stopping the function.")
                        #             result = 300
                        #             p.terminate()
                        #             p.join()
                        #         # Rest of your code here
                        #         else:
                        #             print("Pass timeout criteria.")
                        #             cost_pg, error = queue.get()
                        #             result = cost_pg
                        #         if result == 'NA':
                        #             l_org = ['NA', 'NA', 'NA', 'NA', 'NA']
                        #             break
                        #         elif result == 300:
                        #             l_org = [300, 300, 300, 300, 300]
                        #             break
                        #         else:
                        #             l_org.append(result)
                        #     print(l_org)
                        # else:
                        l_org = [0, 0, 0, 0, 0]
                        # if not all([x == 'NA' for x in l_org[1:]]):
                        #     l_org.sort()
                        #     avg_lat_org = np.mean([x for x in l_org[1:-1] if x != 'NA'])
                        # else:
                        #     avg_lat_org = 'NA'
                        # latency_org.append(avg_lat_org)
                        # if not all([x == 'NA' for x in l_org]):
                        if not all([x == 'NA' for x in l_org]) and latency_org[i] != 300:
                            # if gpt_rules[i] != '[]' and gpt_queries[i] != 'NA' and logical_plan_o != logical_plan_g \
                            #         and org_queries[i].replace(';', '') != gpt_queries[i].replace(';', ''):
                            if gpt_queries[i] != 'NA' and logical_plan_o != logical_plan_g \
                                    and org_queries[i].replace(';', '') != gpt_queries[i].replace(';', ''):
                                print('rewrite!')
                                # same_check = get_execute_sql_detail(org_queries[i], gpt_queries[i], df_id)
                                # print(same_check)
                                # print(latency_org[i])
                                # if same_check:
                                if latency_test[i] != 'NA':
                                    for j in range(5):

                                        queue = multiprocessing.Queue()
                                        p = multiprocessing.Process(target=my_function,
                                                                    args=(db_ids[i], gpt_queries[i], queue))
                                        p.start()
                                        # Set a timeout value
                                        timeout = 300
                                        # Wait for the function to finish or timeout
                                        p.join(timeout)
                                        # Check if the process is still running
                                        if p.is_alive():
                                            print("Function timed out. Stopping the function.")
                                            result = 300
                                            p.terminate()
                                            p.join()
                                        # Rest of your code here
                                        else:
                                            print("Pass timeout criteria.")
                                            cost_pg, error = queue.get()
                                            result = cost_pg
                                        if result == 'NA':
                                            l_gpt = ['NA', 'NA', 'NA', 'NA', 'NA']
                                            break
                                        elif result == 300:
                                            l_gpt = [300, 300, 300, 300, 300]
                                            break
                                        else:
                                            l_gpt.append(result)
                                if not all([x == 'NA' for x in l_gpt]):
                                    l_gpt.sort()
                                    avg_lat_gpt = np.mean([x for x in l_gpt[1:-1] if x != 'NA'])
                                else:
                                    avg_lat_gpt = 'NA'
                                latency_gpt.append(avg_lat_gpt)
                                # else:
                                #     latency_gpt.append('NA')
                            else:
                                # latency_gpt.append(avg_lat_org)
                                latency_gpt.append(latency_org[i])
                        else:
                            latency_gpt.append('NA')

                    if i % 20 == 0:
                        # print(len(latency_org))
                        df_i['db_id'] = ids
                        # df_i['latency_org'] = latency_org
                        df_i['latency_org'] = latency_org[:i + 1]
                        df_i['latency_gpt'] = latency_gpt
                        df_i = pd.DataFrame(df_i)
                        df_i.to_csv('results_baseline_' + df_id + '_1shot_updated_to_' + str(i) + '.csv')
                        # df_i.to_csv('results_' + df_id + '_1shot_' + exp + '_withempty_t0_to_' + str(i) + '.csv')
            results['db_id'] = ids
            results['latency_org'] = latency_org
            results['latency_gpt'] = latency_gpt
            results = pd.DataFrame(results)
            # if jk == 0:
            #     results.to_csv('results_' + df_id + '_1shot_zero.csv')
            # else:
            #     results.to_csv('results_' + df_id + '_1shot_nocl.csv')
            # jk += 1
            # results.to_csv('results_' + df_id + '_1shot_zero.csv')
            results.to_csv('results_baseline_' + df_id + '_1shot_updated.csv')
            # results.to_csv('results_' + df_id + '_1shot_' + exp + '_withempty_t0.csv')
