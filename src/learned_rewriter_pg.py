# @TIME : 24/5/23 4:08 PM
# @AUTHOR : LZDH
import subprocess
import json
import pandas as pd
# from run_postgre import query_cost_postgres
# from run_sqlite import query_cost_sqlite
import re
import time
import multiprocessing


# command = 'ls'
# command = 'cd Rewriter/src'
# db_id = 'browser_web'
# sql_input = "SELECT T1.name FROM browser AS T1 JOIN accelerator_compatible_browser AS T2 ON T1.id  =  T2.browser_id JOIN " \
#             "web_client_accelerator AS T3 ON T2.accelerator_id  =  T3.id WHERE T3.name  =  'CProxy' AND " \
#             "T2.compatible_since_year  >  1998; "


def traverse_lookup_rule(rule_tree, rule_node):
    if rule_tree['cost'] == rule_node['cost']:
        return [rule_tree['activated_rules']]
    if not rule_tree['children']:
        return None
    for child in rule_tree['children']:
        rules_used = traverse_lookup_rule(child, rule_node)
        if rules_used:
            return [rule_tree['activated_rules']] + rules_used
    return None


def call_rewriter(db_id, sql_input):
    # Provide a list of strings as input
    input_list = [db_id, sql_input]
    # Convert the input list to a JSON string
    input_string = json.dumps(input_list)
    command = 'java -cp rewriter_java.jar src/test.java'

    process = subprocess.Popen(command, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True)
    # process.stdin.write(.encode())
    # Wait for the subprocess to finish and capture the output
    output, error = process.communicate(input=input_string)
    # Print the output and error messages
    # print("Output:\n", output)
    # print("Error:\n", error)

    if 'selected' in output:
        output_0 = output.split('selected')[-1]
        if 'new node added..' in output_0:
            output_0 = output_0.split('new node added..')[-1]
        # print(output_0)
        original_node = ' '.join(output_0.split('\n')).split('Original node: ')[1].split('Rewrite node: ')[0][1:]
        original_node = json.loads(original_node)
        text_processed = ' '.join(output_0.split('\n')).split('Original node: ')[1].split('Rewrite node: ')[1]
        if db_id != 'dsb':
            if 'select' in text_processed:
                rewrite_node = text_processed.split('select ')[0]
            else:
                rewrite_node = text_processed.split('SELECT ')[0]
            # print(rewrite_node)
            # rewrite_node = json.loads(rewrite_node)
        else:
            if 'select' in text_processed:
                # print(text_processed)
                if ' with ' in text_processed:
                    rewrite_node = text_processed.split(' with ')[0]
                else:
                    rewrite_node = text_processed.split(' select ')[0]
            else:
                if ' with ' in text_processed:
                    rewrite_node = text_processed.split(' with ')[0]
                else:
                    rewrite_node = text_processed.split(' SELECT ')[0]
            # print(rewrite_node)
        rewrite_node = json.loads(rewrite_node)
        rule_path = [list(i.keys())[0][:-4] for i in traverse_lookup_rule(original_node, rewrite_node) if i]
        rule_path = ['_'.join(re.findall('[A-Z][^A-Z]*', x)).upper() for x in rule_path if rule_path]
    else:
        rule_path = []

    output = output.replace("\u001B[32m", '').replace("\u001B[0m", '').split('\n')
    # print(output)
    ind = 0
    for i in output:
        if not i.startswith('SELECT') and not i.startswith('select') and not i.startswith('with '):
            pass
        else:
            ind = output.index(i)
            break
    queries = output[ind + 1:-3]
    # print(' '.join(queries))
    output = ' '.join(queries).replace('"', '')
    if 'select' in output or 'SELECT' in output or 'Select' in output:
        # change the functions edited to fit calcite back to original ones
        output = output.replace('SUBSTRING', 'SUBSTR')
        return output, rule_path
    else:
        print(db_id)
        print(sql_input)
        print("Output:\n", output)
        print("Error:\n", error)
        if 'INNER' not in error and 'STRFTIME' not in error:
            with open('error_logs_lr_rewrite.txt', 'a+') as f:
                f.write(sql_input)
                f.write('\n')
                f.write(error)
                f.write('\n')
                f.write('\n')
                f.write('\n')
                f.close()
        return 'NA', []


# print(call_rewriter(db_id, sql_input))


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


def my_function(db_id, query, queue):
    result = query_cost_postgres(db_id, query)
    queue.put(result)


if __name__ == '__main__':
    df_LR = {}
    db_ids = []
    original_queries = []
    rewritten_queries = []
    activated_rules = []

    start_time = time.time()
    df_queries = pd.read_csv('data/queries/queries_dsb_test.csv')

    for index, row in df_queries.iterrows():
        # if index > 125 and index != 114 and index != 115:
        if index <= 50:
            print(index)
            df_i = {}
            db_id = row['db_id']
            query = row['original_sql'].replace(';', '') + ';'

            with open('data/schemas_100000/' + db_id + '.json') as f_sch:
                data = f_sch.read()
                schema = json.loads(data)

            # query = query.replace('calcite_', '')
            # query = replace_backticks(query)
            ### to fit for different databases, we have to make edits to the queries

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

            pattern_lim = r'LIMIT\s+(.*?),\s+(.*?)\s+.*'
            matches_lim = re.findall(pattern_lim, query)
            for i in matches_lim:
                query = query.replace('LIMIT ' + i[0] + ', ' + i[1],
                                      'OFFSET ' + i[0] + ' ROWS FETCH NEXT ' + i[1] + ' ROWS ONLY')

            pattern_len = r'LENGTH\((.*?)\)'
            matches_len = re.findall(pattern_len, query)
            for i in matches_len:
                query = query.replace('LENGTH(' + i + ')', 'CHAR_LENGTH(CAST(' + i + ' AS VARCHAR))')

            rewrite_query, rule_path = call_rewriter(db_id, query)

            db_ids.append(db_id)
            query = query.replace('calcite_', '')
            rewrite_query = rewrite_query.replace('calcite_', '')
            fill_list = fill_quotes_list(query)
            for i in fill_list:
                if '"' + i + '"' not in rewrite_query:
                    rewrite_query = rewrite_query.replace(i, '"' + i + '"')
            print(query)
            print(rewrite_query)
            print(rule_path)

            if rewrite_query == 'NA':
                original_queries.append(query)
                rewritten_queries.append('NA')
                activated_rules.append('NA')
                continue

            for i in matches_iif:
                query = query.replace('CASE WHEN ' + i[0] + ' THEN ' + i[1] + ' ELSE ' + i[2] + ' END',
                                      'IIF(' + i[0] + ', ' + i[1] + ', ' + i[2] + ')')
                rewrite_query = rewrite_query.replace('CASE WHEN ' + i[0] + ' THEN ' + i[1] + ' ELSE ' + i[2] + ' END',
                                                      'IIF(' + i[0] + ', ' + i[1] + ', ' + i[2] + ')')
            for i in matches_lim:
                query = query.replace('OFFSET ' + i[0] + ' ROWS FETCH NEXT ' + i[1] + ' ROWS ONLY',
                                      'LIMIT ' + i[0] + ', ' + i[1])
                rewrite_query = rewrite_query.replace('OFFSET ' + i[0] + ' ROWS FETCH NEXT ' + i[1] + ' ROWS ONLY',
                                                      'LIMIT ' + i[0] + ', ' + i[1])

            pattern_lim_2 = r'FETCH NEXT (\d+) ROWS ONLY'
            rewrite_query = re.sub(pattern_lim_2, r'LIMIT \1', rewrite_query)

            for i in matches_len:
                query = query.replace('CHAR_LENGTH(CAST(' + i + ' AS VARCHAR))', 'LENGTH(' + i + ')')
                rewrite_query = rewrite_query.replace('CHAR_LENGTH(CAST(' + i + ' AS VARCHAR))',
                                                      'LENGTH(' + i + ')')

            query = query.replace('CHAR', 'TEXT')
            rewrite_query = rewrite_query.replace('$', '')

            original_queries.append(query)
            rewritten_queries.append(rewrite_query)
            if rule_path:
                activated_rules.append(rule_path)
            else:
                activated_rules.append([])

            # if index % 500 == 0:
            #     df_i['db_id'] = db_ids
            #     df_i['original_sql'] = original_queries
            #     df_i['rewritten_sql'] = rewritten_queries
            #     df_i['activated_rules'] = activated_rules
            #     df_i = pd.DataFrame(df_i)
            #     df_i.to_csv('LR_dsb_results_to_' + str(index) + '.csv')
    end_time = time.time()
    print(end_time - start_time)
    # df_LR['db_id'] = db_ids
    # df_LR['original_sql'] = original_queries
    # df_LR['rewritten_sql'] = rewritten_queries
    # df_LR['activated_rules'] = activated_rules
    #
    # df_LR = pd.DataFrame(df_LR)
    # df_LR.to_csv('LR_dsb_results.csv')
