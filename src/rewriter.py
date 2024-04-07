# @TIME : 24/5/23 4:08 PM
# @AUTHOR : LZDH
import subprocess
import json
import numpy as np

# command = 'ls'
# command = 'cd Rewriter/src'

# example 1 (?)
# sql_input = "SELECT MAX(distinct l_orderkey) FROM lineitem where exists( SELECT MAX(c_custkey) FROM customer where c_custkey = l_orderkey GROUP BY c_custkey )";
# rule_input = ['Aggregate_project_merge'.upper(), 'Project_Merge'.upper()]
# example 2 (good)
# sql_input = "SELECT (SELECT o_orderdate FROM orders limit 1 offset 6 ) AS c0 from region as ref_0 where false limit 76"
# rule_input = ['Sort_Project_Transpose'.upper(), 'Project_To_Calc'.upper()]
# example 3 (good)
# sql_input = "select distinct l_orderkey, sum(l_extendedprice + 3 + (1 - l_discount)) as revenue, o_orderkey, o_shippriority from customer, orders, lineitem where c_mktsegment = 'BUILDING' and c_custkey = o_custkey and l_orderkey = o_orderkey and o_orderdate < date '1995-03-15' and l_shipdate > date '1995-03-15' group by l_orderkey, o_orderkey, o_shippriority order by revenue desc, o_orderkey"
# rule_input = ['Filter_Into_Join'.upper(), 'Project_To_Calc'.upper(), 'Join_extract_Filter'.upper()]
# example 4
# sql_input = "SELECT DISTINCT COUNT ( t3.paperid ) FROM venue AS t4 JOIN paper AS t3 ON t4.venueid  =  t3.venueid JOIN writes AS t2 ON t2.paperid  =  t3.paperid JOIN author AS t1 ON t2.authorid  =  t1.authorid WHERE t1.authorname  =  'David M. Blei' AND t4.venuename  =  'AISTATS'"
# rule_input = ['JOIN_EXTRACT_FILTER'.upper(), 'FILTER_INTO_JOIN'.upper(), 'PROJECT_TO_CALC']
# example 5
db_id = 'tpch'
sql_input = "select l_shipmode, sum(case when o_orderpriority = '1-URGENT' or o_orderpriority = '2-HIGH' then 1 else " \
            "0 end) as high_line_count, sum(case when o_orderpriority <> '1-URGENT' and o_orderpriority <> '2-HIGH' " \
            "then 1 else 0 end) as low_line_count from orders, lineitem where o_orderkey = l_orderkey and l_shipmode " \
            "in ('SHIP', 'MAIL') and l_commitdate < l_receiptdate and l_shipdate < l_commitdate and l_receiptdate >= " \
            "date '1996-01-01' and l_receiptdate < date '1996-01-01' + interval '1' year group by l_shipmode order " \
            "by l_shipmode;"

rule_input = ['PROJECT_TO_CALC']
# rules = ['JOIN_CONDITION_PUSH', 'FILTER_MERGE', 'FILTER_INTO_JOIN', 'JOIN_PROJECT_BOTH_TRANSPOSE', 'PROJECT_MERGE',
#          'JOIN_REDUCE_EXPRESSIONS', 'SORT_JOIN_TRANSPOSE', 'SORT_PROJECT_TRANSPOSE', 'SORT_REMOVE_CONSTANT_KEYS',
#          'SORT_REMOVE']


def call_rewriter(db_id, sql_input, rule_input):
    # Provide a list of strings as input
    input_list = [db_id, sql_input, rule_input]
    # Convert the input list to a JSON string
    input_string = json.dumps(input_list)
    command = 'java -cp rewriter_java.jar src/rule_rewriter.java'

    process = subprocess.Popen(command, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True)
    # process.stdin.write(.encode())
    # Wait for the subprocess to finish and capture the output
    output, error = process.communicate(input=input_string)

    # Print the output and error messages
    # print("Output:\n", output)
    # print("Error:\n", error)
    output = output.replace("\u001B[32m", '').replace("\u001B[0m", '').split('\n')
    ind = 0
    for i in output:
        if not i.startswith('SELECT') and not i.startswith('select') and not i.startswith('with '):
            pass
        else:
            ind = output.index(i)
            break
    queries = output[ind+1:-3]
    # print(' '.join(queries))
    output = ' '.join(queries).replace('"', '')
    if 'select' in output or 'SELECT' in output or 'Select' in output:
        # change the functions edited to fit calcite back to original ones
        output = output.replace('SUBSTRING', 'SUBSTR')
        return output
    else:
        print(db_id)
        print(sql_input)
        print("Output:\n", output)
        print("Error:\n", error)
        with open('error_logs_gpt_rewrite.txt', 'a+') as f:
            f.write(sql_input)
            f.write('\n')
            f.write(error)
            f.write('\n')
            f.write('\n')
            f.write('\n')
            f.close()
        return 'NA'



# def create_nested_tree(heights, nodes, filt_meta):
#     if len(heights) <= 3:
#         if filt_meta:
#             return [i[:i.index('(')] for i in nodes]
#         else:
#             return nodes
#     else:
#         if filt_meta:
#             root = nodes[0][:nodes[0].index('(')]
#         else:
#             root = nodes[0]
#         root_h = heights[0]
#         direct_subs = [i for i, x in enumerate(heights) if x == root_h+1]
#         if len(direct_subs) == 2:
#             left_root = [i for i, x in enumerate(heights) if x == root_h+1][0]
#             right_root = [i for i, x in enumerate(heights) if x == root_h+1][1]
#             left_subtree = create_nested_tree(heights[left_root:right_root], nodes[left_root:right_root])
#             right_subtree = create_nested_tree(heights[right_root:], nodes[right_root:])
#             return [root, left_subtree, right_subtree]
#         else:
#             left_root = [i for i, x in enumerate(heights) if x == root_h+1][0]
#             left_subtree = create_nested_tree(heights[left_root:], nodes[left_root:])
#             return [root, left_subtree]


print(call_rewriter(db_id, sql_input, rule_input))

