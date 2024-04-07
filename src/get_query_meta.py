# @TIME : 12/7/23 5:33 PM
# @AUTHOR : LZDH
import subprocess
import json
# import numpy as np

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
# db_id = 'tpch'
# sql_input = "select ps_partkey, sum(ps_supplycost * ps_availqty) as calcite_value from partsupp, supplier, nation where ps_suppkey = s_suppkey and s_nationkey = n_nationkey and n_name = 'PERU' group by ps_partkey having sum(ps_supplycost * ps_availqty) > ( select sum(ps_supplycost * ps_availqty) * 0.0001000000 from partsupp, supplier, nation where ps_suppkey = s_suppkey and s_nationkey = n_nationkey and n_name = 'PERU' ) order by calcite_value desc;;"
# sql_input_1 = "select ps_partkey, sum(ps_supplycost * ps_availqty) as calcite_value from partsupp, supplier, nation where ps_suppkey = s_suppkey and s_nationkey = n_nationkey and n_name = 'AKSJ' group by ps_partkey having sum(ps_supplycost * ps_availqty) > ( select sum(ps_supplycost * ps_availqty) * 0.01000000 from partsupp, supplier, nation where ps_suppkey = s_suppkey and s_nationkey = n_nationkey and n_name = 'AKSJ' ) order by calcite_value desc;;"

# sql_input = "SELECT t1.ps_partkey, t1.value FROM (SELECT partsupp.ps_partkey, SUM(partsupp.ps_supplycost * partsupp.ps_availqty) AS value FROM partsupp, supplier, nation WHERE partsupp.ps_suppkey = supplier.s_suppkey AND supplier.s_nationkey = nation.n_nationkey AND nation.n_name = 'PERU' GROUP BY partsupp.ps_partkey) AS t1 LEFT JOIN (SELECT SUM(partsupp0.ps_supplycost * partsupp0.ps_availqty) * 0.0001000000 AS EXPR0 FROM partsupp AS partsupp0, supplier AS supplier0, nation AS nation0 WHERE partsupp0.ps_suppkey = supplier0.s_suppkey AND supplier0.s_nationkey = nation0.n_nationkey AND nation0.n_name = 'PERU') AS t5 ON TRUE WHERE t1.value > t5.EXPR0 ORDER BY t1.value DESC;"

# node_str = "LogicalFilter(condition=[AND(=($7, 'CProxy'), >($5, 1998))]): rowcount = 4.221408232700325E11, cumulative cost = {6.051060523576132E12 rows, 5.6285445353031E12 cpu, 0.0 io}, id = 17"

reps = ["year", "date", "rank", "position", "YEAR", "DATE", "RANK", "POSITION", "Year", "Date", "Rank", "Position",
        "TIME", "Time", "time", "KEY", "Key", "key", "DAY", "Day", "day", "PER", "Per", "per", "RESULT", "Result",
        "result", "MONTH", "Month", "month", "METHOD", "Method", "method", "RATING", "Rating", "rating", "CHARACTER",
        "Character", "RANGE", "Range", "range", "count"]


def process_plan_node(node_str, row_only):
    operator = node_str[:node_str.index('(')]
    operator_detail = node_str[node_str.index('('):node_str.index(': rowcount')]
    row_count = node_str.split('rowcount = ')[1].split(',')[0]
    cum_costs = node_str.split('cumulative cost = ')[1].split('}')[0][1:]
    cum_rows = cum_costs.split(' rows, ')[0]
    cum_cpu = cum_costs.split(' rows, ')[1].split(' cpu, ')[0]
    if row_only:
        plan_node = operator + '(' + str(row_count) + ' rows)'
    else:
        plan_node = {'NodeType': operator,
                     'Node_Detail': operator_detail,
                     'Rows': float(row_count),
                     'Cum_Rows': float(cum_rows),
                     'Cum_Cpu': float(cum_cpu)}

    return plan_node


def create_nested_tree(heights, nodes, filt_meta, row_only=False):
    if len(heights) <= 3:
        if filt_meta:
            return [i[:i.index('(')] for i in nodes]
        else:
            # print(row_only)
            return [process_plan_node(i, row_only) for i in nodes]
    else:
        if filt_meta:
            root = nodes[0][:nodes[0].index('(')]
        else:
            # print(row_only)
            root = process_plan_node(nodes[0], row_only)
        root_h = heights[0]
        direct_subs = [i for i, x in enumerate(heights) if x == root_h + 1]
        if len(direct_subs) == 2:
            left_root = [i for i, x in enumerate(heights) if x == root_h + 1][0]
            right_root = [i for i, x in enumerate(heights) if x == root_h + 1][1]
            left_subtree = create_nested_tree(heights[left_root:right_root], nodes[left_root:right_root], filt_meta, row_only)
            right_subtree = create_nested_tree(heights[right_root:], nodes[right_root:], filt_meta, row_only)
            return [root, left_subtree, right_subtree]
        else:
            left_root = [i for i, x in enumerate(heights) if x == root_h + 1][0]
            left_subtree = create_nested_tree(heights[left_root:], nodes[left_root:], filt_meta, row_only)
            return [root, left_subtree]


def get_logical_plan(db_id, sql_input):
    for s in reps:
        s = ' ' + s + ' '
        sql_input.replace(s, "'" + s + "'")
    # Provide a list of strings as input
    input_list = [db_id, sql_input]
    # Convert the input list to a JSON string
    input_string = json.dumps(input_list)
    command = 'java -cp rewriter_java.jar src/get_logical_plan.java'

    process = subprocess.Popen(command, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True)
    output, error = process.communicate(input=input_string)

    # Print the output and error messages
    # print("Output:\n", output)
    # print(sql_input)
    # print("Error:\n", error)
    output = output.replace("\u001B[32m", '').replace("\u001B[0m", '').split('\n')
    # print(output)
    ind = 0
    for i in output:
        if 'successfully.' in i:
            pass
        else:
            ind = output.index(i)
            break
    output = output[ind:-2]
    nodes = [i.replace('  ', '').replace('    ', '') for i in output]
    heights = [(len(s) - len(s.lstrip())) // 2 for s in output]
    if nodes:
        if '(' in nodes[0]:
            plan_clean = create_nested_tree(heights, nodes, True)
        else:
            plan_clean = []
    else:
        plan_clean = []
    return plan_clean


def get_physical_tree(db_id, sql_input, row_only=False):
    for s in reps:
        s = ' ' + s + ' '
        sql_input.replace(s, "'" + s + "'")
    # Provide a list of strings as input
    input_list = [db_id, sql_input]
    # Convert the input list to a JSON string
    input_string = json.dumps(input_list)
    command = 'java -cp rewriter_java.jar src/get_physical_tree.java'

    process = subprocess.Popen(command, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True)
    output, error = process.communicate(input=input_string)

    # Print the output and error messages
    # print("Output:\n", output)
    # print(sql_input)
    # print("Error:\n", error)
    output = output.replace("\u001B[32m", '').replace("\u001B[0m", '').split('\n')
    ind = 0
    for i in output:
        if 'successfully.' in i:
            pass
        else:
            ind = output.index(i)
            break
    output = output[ind:-2]
    nodes = [i.replace('  ', '').replace('    ', '') for i in output]
    heights = [(len(s) - len(s.lstrip())) // 2 for s in output]
    if nodes:
        if '(' in nodes[0]:
            physical_plan = create_nested_tree(heights, nodes, False, row_only)
        else:
            physical_plan = []
    else:
        physical_plan = []
    return physical_plan


# db_id = 'dsb'
# sql_input = "with my_customers as ( select distinct c_customer_sk  , c_current_addr_sk from  ( select cs_sold_date_sk sold_date_sk,  cs_bill_customer_sk customer_sk,  cs_item_sk item_sk,  cs_wholesale_cost wholesale_cost from catalog_sales union all select ws_sold_date_sk sold_date_sk,  ws_bill_customer_sk customer_sk,  ws_item_sk item_sk,  ws_wholesale_cost wholesale_cost from web_sales ) cs_or_ws_sales, item, date_dim, customer where sold_date_sk = d_date_sk and item_sk = i_item_sk and i_category = 'Men' and i_class = 'sports-apparel' and c_customer_sk = cs_or_ws_sales.customer_sk and d_moy = 8 and d_year = 2000 and wholesale_cost BETWEEN 59 AND 89 and c_birth_year BETWEEN 1936 AND 1949 ) , my_revenue as ( select c_customer_sk,  sum(ss_ext_sales_price) as revenue from my_customers,  store_sales,  customer_address,  store,  date_dim where c_current_addr_sk = ca_address_sk  and ca_county = s_county  and ca_state = s_state  and ss_sold_date_sk = d_date_sk  and c_customer_sk = ss_customer_sk  and ss_wholesale_cost BETWEEN 59 AND 89  and s_state in ('IL','LA','MO'   ,'MS','NJ','OH'   ,'SD','TN','VA'   ,'WY')  and d_month_seq between (select distinct d_month_seq+1   from date_dim where d_year = 2000 and d_moy = 8)   and (select distinct d_month_seq+3   from date_dim where d_year = 2000 and d_moy = 8) group by c_customer_sk ) , segments as (select cast((revenue/50) as int) as segment from my_revenue ) select segment, count(*) as num_customers, segment*50 as segment_base from segments group by segment order by segment, num_customers limit 100 ;"
# print(get_logical_plan(db_id, sql_input))
# print(get_physical_tree(db_id, sql_input))

