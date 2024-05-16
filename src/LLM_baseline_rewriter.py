# @TIME : 6/2/24 7:30 PM
# @AUTHOR : LZDH
import pandas as pd
import re
import openai
import time

from encoder import *

from openai import OpenAI
# import tiktoken
client = OpenAI(
    # This is the default and can be omitted
    api_key="your_api_key"
)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def query_gpt_attempts(prompt, trys):
    try:
        output = query_turbo_model(prompt)
    except:
        print(trys)
        trys += 1
        if trys <= 3:
            output = query_gpt_attempts(prompt, trys)
        else:
            output = {'content': 'NA'}
    return output


def query_turbo_model(prompt):
    chat_completion = client.chat.completions.create(
        messages=prompt,
        model="gpt-3.5-turbo",
        temperature=0,
    )
    # print(chat_completion)
    return chat_completion.choices[0].message.content
    # completion = openai.ChatCompletion.create(
    #     model="gpt-4-1106-preview",
    #     messages=prompt,
    #     temperature=0
    # )
    # return completion['choices'][0]['message']


def fill_quotes_list(original_sql):
    fill_list = []
    count = 0
    for i in range(len(original_sql)):
        if i != 0 and i != len(original_sql) - 1:
            char = original_sql[i]
            if char == '"':
                count += 1
                if count % 2 == 1:
                    start_ind = i
                else:
                    end_ind = i
                    seg = original_sql[start_ind: end_ind].replace('"', '')
                    fill_list.append(seg)
    # print(fill_list)
    return fill_list


def filter_gpt_output(gpt_output):
    if gpt_output == 'NA':
        return 'NA'
    # print(gpt_output)
    out_sql = gpt_output['content'].split('[')[-1].split(']')[0].split(':')[-1]
    out_sql = out_sql.replace('/', '').replace('"', '')
    if out_sql[0] == ' ':
        out_sql = out_sql[1:]
    print('out_sql: ', out_sql)
    return out_sql


def generate_turbo_prompt_one(schema, query, promotions):
    schema_0, query_0, rewrite_0 = promotions[0]
    p = [{'role': "system", 'content': 'You are an online SQL rewrite agent. You will be given a schema of a dataset'
                                       ' and a SQL query based on this schema. You are required to'
                                       ' rewrite the query to improve the efficiency of running this query. '
                                       'You should return only the rewritten query, in the '
                                       'form of "New query: [the rewritten query]". '},
         {
             'role': "user",
             'content': "Schema: " + str(schema_0) + ". Query: " + str(query_0) + ".",
         },
         {
             'role': "assistant",
             'content': str(rewrite_0),
         },
         {
             'role': "user",
             'content': "Schema: " + str(schema) + ". Query: " + str(query) + ".",
         }, ]
    return p


schema_0 = '[{"table":"customer","rows":1500000,"columns":[{"name":"c_custkey","type":"integer"},{"name":"c_name","type":"character varying"},{"name":"c_address","type":"character varying"},{"name":"c_nationkey","type":"integer"},{"name":"c_phone","type":"character"},{"name":"c_acctbal","type":"numeric"},{"name":"c_mktsegment","type":"character"},{"name":"c_comment","type":"character varying"}]},{"table":"lineitem","rows":60000000,"columns":[{"name":"l_orderkey","type":"integer"},{"name":"l_partkey","type":"integer"},{"name":"l_suppkey","type":"integer"},{"name":"l_linenumber","type":"integer"},{"name":"l_quantity","type":"numeric"},{"name":"l_extendedprice","type":"numeric"},{"name":"l_discount","type":"numeric"},{"name":"l_tax","type":"numeric"},{"name":"l_returnflag","type":"character"},{"name":"l_linestatus","type":"character"},{"name":"l_shipdate","type":"date"},{"name":"l_commitdate","type":"date"},{"name":"l_receiptdate","type":"date"},{"name":"l_shipinstruct","type":"character"},{"name":"l_shipmode","type":"character"},{"name":"l_comment","type":"character varying"}]},{"table":"nation","rows":25,"columns":[{"name":"n_nationkey","type":"integer"},{"name":"n_name","type":"character"},{"name":"n_regionkey","type":"integer"},{"name":"n_comment","type":"character varying"}]},{"table":"orders","rows":60000000,"columns":[{"name":"o_orderkey","type":"integer"},{"name":"o_custkey","type":"integer"},{"name":"o_orderstatus","type":"character"},{"name":"o_totalprice","type":"numeric"},{"name":"o_orderdate","type":"date"},{"name":"o_orderpriority","type":"character"},{"name":"o_clerk","type":"character"},{"name":"o_shippriority","type":"integer"},{"name":"o_comment","type":"character varying"}]},{"table":"part","rows":2000000,"columns":[{"name":"p_partkey","type":"integer"},{"name":"p_name","type":"character varying"},{"name":"p_mfgr","type":"character"},{"name":"p_brand","type":"character"},{"name":"p_type","type":"character varying"},{"name":"p_size","type":"integer"},{"name":"p_container","type":"character"},{"name":"p_retailprice","type":"numeric"},{"name":"p_comment","type":"character varying"}]},{"table":"partsupp","rows":8000000,"columns":[{"name":"ps_partkey","type":"integer"},{"name":"ps_suppkey","type":"integer"},{"name":"ps_availqty","type":"integer"},{"name":"ps_supplycost","type":"numeric"},{"name":"ps_comment","type":"character varying"}]},{"table":"region","rows":5,"columns":[{"name":"r_regionkey","type":"integer"},{"name":"r_name","type":"character"},{"name":"r_comment","type":"character varying"}]},{"table":"supplier","rows":100000,"columns":[{"name":"s_suppkey","type":"integer"},{"name":"s_name","type":"character"},{"name":"s_address","type":"character varying"},{"name":"s_nationkey","type":"integer"},{"name":"s_phone","type":"character"},{"name":"s_acctbal","type":"numeric"},{"name":"s_comment","type":"character varying"}]}]'
query_0 = "select l_orderkey, sum(l_extendedprice * (1 - l_discount)) as revenue, o_orderdate, o_shippriority from customer, orders, lineitem where c_mktsegment = 'AUTOMOBILE' and c_custkey = o_custkey and l_orderkey = o_orderkey and o_orderdate < date '1995-03-18' and l_shipdate > date '1995-03-18' group by l_orderkey, o_orderdate, o_shippriority order by revenue desc, o_orderdate;"
rewrite_0 = "SELECT t1.l_orderkey, SUM(t1.l_extendedprice * (1 - t1.l_discount)) AS revenue, t0.o_orderdate, t0.o_shippriority FROM (SELECT * FROM customer WHERE c_mktsegment = 'AUTOMOBILE') AS t INNER JOIN (SELECT * FROM orders WHERE o_orderdate < DATE '1995-03-18') AS t0 ON t.c_custkey = t0.o_custkey INNER JOIN (SELECT * FROM lineitem WHERE l_shipdate > DATE '1995-03-18') AS t1 ON t0.o_orderkey = t1.l_orderkey GROUP BY t1.l_orderkey, t0.o_orderdate, t0.o_shippriority ORDER BY SUM(t1.l_extendedprice * (1 - t1.l_discount)) DESC, t0.o_orderdate"
datasets = ['dsb', 'job_syn', 'tpch']
num_promos = 1
# methods = ['sentbert']
for dataset in datasets:
    demo_time_record = []
    llm_time_record = []
    rewriter_time_record = []

    df_gpt = {}
    db_ids = []
    original_queries = []
    rewritten_queries_s = []
    df_test = pd.read_csv('data/queries_' + dataset + '_test.csv').fillna('NA')
    for index, row in df_test.iterrows():
        if index >= 0:
            df_i = {}
            db_id = row['db_id']
            db_ids.append(db_id)
            query = row['original_sql'].replace(';', '') + ';'

            if query == 'NA;':
                original_queries.append('NA')
                rewritten_queries_s.append('NA')
            else:
                with open('data/schemas_100000/' + db_id + '.json') as f_sch:
                    data = f_sch.read()
                    schema = json.loads(data)

                filtered_schema = []
                q_names = query.split()
                for tab in schema:
                    if tab['table'] in q_names or (',' + tab['table']) in q_names or (tab['table'] + ',') in q_names:
                        new_tab = {'table': tab['table'], 'rows': tab['rows'], 'columns': []}
                        for col in tab['columns']:
                            if col['name'] in q_names or (',' + col['name']) in q_names or (col['name'] + ',') in q_names:
                                new_tab['columns'].append(col)
                        filtered_schema.append(new_tab)

                schema = filtered_schema

                query = query.replace('`', '"')
                query = query.replace('TEXT', 'CHAR')

                pattern_iif = r'IIF\((.*?),\s+(.*?),\s+(.*?)\)'
                matches_iif = re.findall(pattern_iif, query)
                for i in matches_iif:
                    query = query.replace('IIF(' + i[0] + ', ' + i[1] + ', ' + i[2] + ')',
                                          'CASE WHEN ' + i[0] + ' THEN ' + i[1] + ' ELSE ' + i[2] + ' END')

                # print(query)
                pattern_lim = r'LIMIT\s+(.*?),\s+(.*?)\s+.*'
                matches_lim = re.findall(pattern_lim, query)
                for i in matches_lim:
                    query = query.replace('LIMIT ' + i[0] + ', ' + i[1],
                                          'OFFSET ' + i[0] + ' ROWS FETCH NEXT ' + i[1] + ' ROWS ONLY')

                pattern_len = r'LENGTH\((.*?)\)'
                matches_len = re.findall(pattern_len, query)
                for i in matches_len:
                    query = query.replace('LENGTH(' + i + ')', 'CHAR_LENGTH(CAST(' + i + ' AS VARCHAR))')

                sim_promos = [[schema_0, query_0, rewrite_0]]
                sim_prompt = generate_turbo_prompt_one(schema, query, sim_promos)

                llm_time_start = time.time()
                trys = 0
                gpt_output_s = query_gpt_attempts(sim_prompt, trys)
                rewrite_query_s = filter_gpt_output(gpt_output_s)

                llm_time_end = time.time()
                llm_time = llm_time_end - llm_time_start
                llm_time_record.append(llm_time)

                query = query.replace('calcite_', '')
                rewrite_query_s = rewrite_query_s.replace('calcite_', '')
                fill_list = fill_quotes_list(query)
                for i in fill_list:
                    if '"' + i + '"' not in rewrite_query_s:
                        rewrite_query_s = rewrite_query_s.replace(i, '"' + i + '"')

                # edit some syntax for sqlite
                for i in matches_iif:
                    query = query.replace('CASE WHEN ' + i[0] + ' THEN ' + i[1] + ' ELSE ' + i[2] + ' END',
                                          'IIF(' + i[0] + ', ' + i[1] + ', ' + i[2] + ')')
                    rewrite_query_s = rewrite_query_s.replace(
                        'CASE WHEN ' + i[0] + ' THEN ' + i[1] + ' ELSE ' + i[2] + ' END',
                        'IIF(' + i[0] + ', ' + i[1] + ', ' + i[2] + ')')
                for i in matches_lim:
                    query = query.replace('OFFSET ' + i[0] + ' ROWS FETCH NEXT ' + i[1] + ' ROWS ONLY',
                                          'LIMIT ' + i[0] + ', ' + i[1])
                    rewrite_query_s = rewrite_query_s.replace('OFFSET ' + i[0] + ' ROWS FETCH NEXT ' + i[1] + ' ROWS ONLY',
                                                              'LIMIT ' + i[0] + ', ' + i[1])
                pattern_lim_2 = r'FETCH NEXT (\d+) ROWS ONLY'
                rewrite_query_s = re.sub(pattern_lim_2, r'LIMIT \1', rewrite_query_s)
                for i in matches_len:
                    query = query.replace('CHAR_LENGTH(CAST(' + i + ' AS VARCHAR))', 'LENGTH(' + i + ')')
                    rewrite_query_s = rewrite_query_s.replace('CHAR_LENGTH(CAST(' + i + ' AS VARCHAR))',
                                                              'LENGTH(' + i + ')')
                query = query.replace('CHAR', 'TEXT')
                rewrite_query_s = rewrite_query_s.replace('$', '')

                original_queries.append(query)

    df_gpt['db_id'] = db_ids
    df_gpt['original_sql'] = original_queries
    df_gpt['rewritten_sql_gpt'] = rewritten_queries_s
    df_gpt = pd.DataFrame(df_gpt)
    df_gpt.to_csv('gpt_baseline_' + dataset + '_one_promo_updated.csv')

    df_t = {'llm_time': llm_time_record}
    df_t = pd.DataFrame(df_t)
    df_t.to_csv('time_gpt_baseline_' + dataset + '_one_promo_updated.csv')
