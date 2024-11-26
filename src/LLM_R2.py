# @TIME : 5/6/23 3:53 PM
# @AUTHOR : LZDH
# @TIME : 9/5/23 4:07 PM
# @AUTHOR : LZDH
import random
import pandas as pd
import re
# import openai
import zss
import ast
import time
from rewriter import *
from get_query_meta import *
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import sys

# Set PATHs
PATH_TO_SENTEVAL = './SentEval'
PATH_TO_DATA = './SentEval/data'
# Import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval
from senteval.utils import cosine
from encoder import *
from models import QueryformerForCL
from openai import OpenAI
# import tiktoken
client = OpenAI(
    # This is the default and can be omitted
    api_key="your_openai_api_key"
)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
pre_lang_model = SentenceTransformer('all-MiniLM-L6-v2')

model = QueryformerForCL()
model_name = 'tpch'
checkpoint = torch.load('simcse_models/' + model_name + '/pytorch_model.bin', map_location=torch.device('cpu'))
model.load_state_dict(checkpoint, strict=False)
model.eval()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)


def batcher(sentences, db_ids):
    # sentences = [[' '.join(s).replace('"', '')] for s in batch]
    # db_ids = ['tpch'] * len(sentences)
    sent_features = prepare_enc_data(sentences, pre_lang_model, db_ids)
    batch = eval_collator(sent_features)
    with torch.no_grad():
        outputs = model(**batch, eval=True)
        # pooler_output = outputs.hidden_states
        pooler_output = outputs
    return pooler_output.cpu()


compute_similarity = lambda s1, s2: np.nan_to_num(cosine(np.nan_to_num(s1), np.nan_to_num(s2)))


def query_gpt_attempts(prompt, trys):
    # output = query_turbo_model(prompt)
    try:
        output = query_turbo_model(prompt)
    except:
        print(trys)
        trys += 1
        if trys <= 3:
            output = query_gpt_attempts(prompt, trys)
        else:
            output = 'NA'
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


agge_rewrite_rules = '["AGGREGATE_EXPAND_DISTINCT_AGGREGATES": "Rule that expands distinct aggregates (such as COUNT(DISTINCT x)) from a Aggregate"], ' \
                     '["AGGREGATE_EXPAND_DISTINCT_AGGREGATES_TO_JOIN": "As AGGREGATE_EXPAND_DISTINCT_AGGREGATES but generates a Join"], ' \
                     '["AGGREGATE_JOIN_TRANSPOSE_EXTENDED"： "As AGGREGATE_JOIN_TRANSPOSE, but extended to push down aggregate functions"], ' \
                     '["AGGREGATE_PROJECT_MERGE": "Rule that recognizes an Aggregate on top of a Project and if possible aggregates through the Project or removes the Project"], ' \
                     '["AGGREGATE_ANY_PULL_UP_CONSTANTS": "More general form of AGGREGATE_PROJECT_PULL_UP_CONSTANTS that matches any relational expression"], ' \
                     '["AGGREGATE_UNION_AGGREGATE": "Rule that matches an Aggregate whose input is a Union one of whose inputs is an Aggregate"], ' \
                     '["AGGREGATE_UNION_TRANSPOSE": "Rule that pushes an Aggregate past a non-distinct Union"], ' \
                     '["AGGREGATE_VALUES": "Rule that applies an Aggregate to a Values (currently just an empty Values)"], ' \
                     '["AGGREGATE_REMOVE": "Rule that removes an Aggregate if it computes no aggregate functions (that is, it is implementing SELECT DISTINCT), or all the aggregate functions are splittable, and the underlying relational expression is already distinct"], '

filt_rewrite_rules = '["FILTER_AGGREGATE_TRANSPOSE": "Rule that pushes a Filter past an Aggregate"], ' \
                     '["FILTER_CORRELATE": "Rule that pushes a Filter above a Correlate into the inputs of the Correlate"], ' \
                     '["FILTER_INTO_JOIN": "Rule that tries to push filter expressions into a join condition and into the inputs of the join"], ' \
                     '["JOIN_CONDITION_PUSH": "Rule that pushes predicates in a Join into the inputs to the join"], ' \
                     '["FILTER_MERGE": "Rule that combines two LogicalFilters"], ' \
                     '["FILTER_MULTI_JOIN_MERGE": "Rule that merges a Filter into a MultiJoin, creating a richer MultiJoin"], ' \
                     '["FILTER_PROJECT_TRANSPOSE": "The default instance of FilterProjectTransposeRule"], ' \
                     '["FILTER_SET_OP_TRANSPOSE": "Rule that pushes a Filter past a SetOp"], ' \
                     '["FILTER_TABLE_FUNCTION_TRANSPOSE": "Rule that pushes a LogicalFilter past a LogicalTableFunctionScan"], ' \
                     '["FILTER_SCAN": "Rule that matches a Filter on a TableScan"], ' \
                     '["FILTER_REDUCE_EXPRESSIONS": "Rule that reduces constants inside a LogicalFilter"], ' \
                     '["PROJECT_REDUCE_EXPRESSIONS": "Rule that reduces constants inside a LogicalProject"], '

join_rewrite_rules = '["JOIN_EXTRACT_FILTER": "Rule to convert an inner join to a filter on top of a cartesian inner join": ], ' \
                     '["JOIN_PROJECT_BOTH_TRANSPOSE": "Rule that matches a LogicalJoin whose inputs are LogicalProjects, and pulls the project expressions up"], ' \
                     '["JOIN_PROJECT_LEFT_TRANSPOSE": "As JOIN_PROJECT_BOTH_TRANSPOSE but only the left input is a LogicalProject"], ' \
                     '["JOIN_PROJECT_RIGHT_TRANSPOSE": "As JOIN_PROJECT_BOTH_TRANSPOSE but only the right input is a LogicalProject"], ' \
                     '["JOIN_LEFT_UNION_TRANSPOSE": "Rule that pushes a Join past a non-distinct Union as its left input"], ' \
                     '["JOIN_RIGHT_UNION_TRANSPOSE": "Rule that pushes a Join past a non-distinct Union as its right input"], ' \
                     '["SEMI_JOIN_REMOVE": "Rule that removes a semi-join from a join tree"], ' \
                     '["JOIN_REDUCE_EXPRESSIONS": "Rule that reduces constants inside a Join"], '

sort_rewrite_rules = '["SORT_JOIN_TRANSPOSE": "Rule that pushes a Sort past a Join"], ' \
                     '["SORT_PROJECT_TRANSPOSE": "Rule that pushes a Sort past a Project"], ' \
                     '["SORT_UNION_TRANSPOSE": "Rule that pushes a Sort past a Union"], ' \
                     '["SORT_REMOVE_CONSTANT_KEYS": "Rule that removes keys from a Sort if those keys are known to be constant, or removes the entire Sort if all keys are constant"], ' \
                     '["SORT_REMOVE": "Rule that removes a Sort if its input is already sorted"], '

union_rewrite_rules = '["UNION_MERGE": "Rule that flattens a Union on a Union into a single Union"], ' \
                      '["UNION_REMOVE": "Rule that removes a Union if it has only one input"], ' \
                      '["UNION_TO_DISTINCT": "Rule that translates a distinct Union (all = false) into an Aggregate on top of a non-distinct Union (all = true)"], ' \
                      '["UNION_PULL_UP_CONSTANTS": "Rule that pulls up constants through a Union operator"], '


def generate_turbo_prompt_light(schema, query, logical_plan, promotions):
    p = [{'role': "system", 'content': 'You are an online SQL rewrite agent. You will be given a SQL query.'
                                       ' You are required to propose rewriting rules to'
                                       ' rewrite the query to improve the efficiency of running this query, using the '
                                       'given rewriting rules below. The rules are provided in form of ["rule name": '
                                       '"rule description"] and you should answer with a list of rewriting rule names, '
                                       'which if applied in sequence, will best rewrite the input SQL query into a new '
                                       'query, which is the most efficient. '
                                       'Return "Empty List" if from the previous chat and input query, no rule should be used. '
                                       # 'Try to use as few rules as possible and '
                                       # 'return an empty list if you think the SQL query given is already efficient. '
                                       'The rewriting rules you can adopt are defined as follows: ' +
                                       agge_rewrite_rules + filt_rewrite_rules + join_rewrite_rules +
                                       # proj_rewrite_rules + calc_rewrite_rules + prune_empty_rules +
                                       union_rewrite_rules + sort_rewrite_rules +
                                       'You should return only a list of rewriting rule names provided above, in the '
                                       'form of "Rules selected: [rule names]".'}]
    for promo in promotions:
        schema_p, query_p, logical_plan_p, rules_list_p = promo
        print('demo sql: ', str(query_p))
        print('demo rules: Rules selected: ', str(rules_list_p))
        promo_p = [{
            'role': "user",
            'content': "Query: " + str(query_p),
        },
            {
                'role': "assistant",
                'content': 'Rules selected: ' + str(rules_list_p),
            }]
        p = p + promo_p
    p.append({
        'role': "user",
        'content': "Query: " + str(query),
    })
    return p


def generate_llama2_prompt_light(query, promotions):
    p = 'You are an online SQL rewrite agent. You will be given a SQL query. You are required to propose rewriting ' \
        'rules to rewrite the query to improve the efficiency of running this query, using the given rewriting rules ' \
        'below. The rules are provided in form of ["rule name": "rule description"] and you should answer with a ' \
        'list of rewriting rule names, which if applied in sequence, will best rewrite the input SQL query into a ' \
        'new query, which is the most efficient. Return "Empty List" if from the previous chat and input query, no ' \
        'rule should be used. The rewriting rules you can adopt are defined as follows: ' \
        + agge_rewrite_rules + filt_rewrite_rules + join_rewrite_rules + union_rewrite_rules + sort_rewrite_rules + \
        'You should return only a list of rewriting rule names provided above, in the form of ' \
        '"Rules selected: [rule names]".'

    for promo in promotions:
        schema_p, query_p, logical_plan_p, rules_list_p = promo
        print('demo rules: Rules selected: ', str(rules_list_p))
        promo_p = " Query: " + str(query_p) + '. Rules selected: ' + str(rules_list_p) + "."
        p = p + promo_p
    p += " Query: " + str(query) + "."
    return p


def filter_gpt_output(gpt_output):
    rule_list = ['AGGREGATE_EXPAND_DISTINCT_AGGREGATES', 'AGGREGATE_EXPAND_DISTINCT_AGGREGATES_TO_JOIN',
                 'AGGREGATE_JOIN_TRANSPOSE_EXTENDED', 'AGGREGATE_PROJECT_MERGE', 'AGGREGATE_ANY_PULL_UP_CONSTANTS',
                 'AGGREGATE_UNION_AGGREGATE', 'AGGREGATE_UNION_TRANSPOSE', 'AGGREGATE_VALUES', 'AGGREGATE_INSTANCE',
                 'AGGREGATE_REMOVE', 'FILTER_AGGREGATE_TRANSPOSE', 'FILTER_CORRELATE', 'FILTER_INTO_JOIN',
                 'JOIN_CONDITION_PUSH', 'FILTER_MERGE', 'FILTER_MULTI_JOIN_MERGE', 'FILTER_PROJECT_TRANSPOSE',
                 'FILTER_SET_OP_TRANSPOSE', 'FILTER_TABLE_FUNCTION_TRANSPOSE', 'FILTER_SCAN',
                 'FILTER_REDUCE_EXPRESSIONS', 'PROJECT_REDUCE_EXPRESSIONS', 'FILTER_INSTANCE', 'JOIN_EXTRACT_FILTER',
                 'JOIN_PROJECT_BOTH_TRANSPOSE', 'JOIN_PROJECT_LEFT_TRANSPOSE', 'JOIN_PROJECT_RIGHT_TRANSPOSE',
                 'JOIN_LEFT_UNION_TRANSPOSE', 'JOIN_RIGHT_UNION_TRANSPOSE', 'SEMI_JOIN_REMOVE',
                 'JOIN_REDUCE_EXPRESSIONS', 'JOIN_LEFT_INSTANCE', 'JOIN_RIGHT_INSTANCE', 'PROJECT_CALC_MERGE',
                 'PROJECT_CORRELATE_TRANSPOSE', 'PROJECT_MERGE', 'PROJECT_MULTI_JOIN_MERGE', 'PROJECT_REMOVE',
                 'PROJECT_TO_CALC', 'PROJECT_SUB_QUERY_TO_CORRELATE', 'PROJECT_REDUCE_EXPRESSIONS',
                 'PROJECT_INSTANCE', 'CALC_MERGE', 'CALC_REMOVE', 'SORT_JOIN_TRANSPOSE', 'SORT_PROJECT_TRANSPOSE',
                 'SORT_UNION_TRANSPOSE', 'SORT_REMOVE_CONSTANT_KEYS', 'SORT_REMOVE', 'SORT_INSTANCE',
                 'SORT_FETCH_ZERO_INSTANCE', 'UNION_MERGE', 'UNION_REMOVE', 'UNION_TO_DISTINCT',
                 'UNION_PULL_UP_CONSTANT', 'UNION_INSTANCE', 'INTERSECT_INSTANCE', 'MINUS_INSTANCE']
    if gpt_output == 'NA':
        return []
    out_rules = gpt_output.split('[')[-1].split(']')[0]
    out_rules = out_rules.replace('/', '').replace('"', '').replace("'", "")
    out_rules = [x.replace(' ', '').replace('\n', '').strip() for x in out_rules.split(',')]
    print('out_rules: ', out_rules)
    execute_rules = []
    for r in out_rules:
        if r in rule_list:
            execute_rules.append(r)

    return execute_rules


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


def get_promo_meta(db_id, query, rule_path):
    with open('../data/data_llmr2/schemas/' + db_id + '.json') as f_sch:
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
    query_1 = query.replace('`', '"')
    query_1 = query_1.replace('TEXT', 'CHAR')
    pattern_iif = r'IIF\((.*?),\s+(.*?),\s+(.*?)\)'
    matches_iif = re.findall(pattern_iif, query_1)
    for i in matches_iif:
        query_1 = query_1.replace('IIF(' + i[0] + ', ' + i[1] + ', ' + i[2] + ')',
                                  'CASE WHEN ' + i[0] + ' THEN ' + i[1] + ' ELSE ' + i[2] + ' END')
    pattern_lim = r'LIMIT\s+(.*?),\s+(.*?)\s+.*'
    matches_lim = re.findall(pattern_lim, query_1)
    for i in matches_lim:
        query_1 = query_1.replace('LIMIT ' + i[0] + ', ' + i[1],
                                  'OFFSET ' + i[0] + ' ROWS FETCH NEXT ' + i[1] + ' ROWS ONLY')
    pattern_len = r'LENGTH\((.*?)\)'
    matches_len = re.findall(pattern_len, query_1)
    for i in matches_len:
        query_1 = query_1.replace('LENGTH(' + i + ')', 'CHAR_LENGTH(CAST(' + i + ' AS VARCHAR))')

    logical_plan = get_logical_plan(db_id, query_1)
    return schema, query, logical_plan, rule_path


def edit_queries(textsql):
    rep = {" year ": " calcite_year ", " date ": " calcite_date ", " rank ": "calcite_rank ", " position ": " calcite_position ",
           " YEAR ": " calcite_YEAR ", " DATE ": " calcite_DATE ", " RANK ": "calcite_RANK ", " POSITION ": " calcite_POSITION ",
           " Year ": " calcite_Year ", " Date ": " calcite_Date ", " Rank ": "calcite_Rank ", " Position ": " calcite_Position ",
           " TIME ": " calcite_TIME ", " Time ": " calcite_Time ", " time ": "calcite_time ",
           " KEY ": " calcite_KEY ", " Key ": " calcite_Key ", " key ": " calcite_key ",
           " DAY ": " calcite_DAY ", " Day ": " calcite_Day ", " day ": " calcite_day ",
           " PER ": " calcite_PER ", " Per ": " calcite_Per ", " per ": " calcite_per ",
           " RESULT ": " calcite_RESULT ", " Result ": " calcite_Result ", " result ": " calcite_result ",
           " MONTH ": " calcite_MONTH ", " Month ": " calcite_Month ", " month ": " calcite_month ",
           " METHOD ": " calcite_METHOD ", " Method ": " calcite_Method ", " method ": " calcite_method ",
           " RATING ": " calcite_RATING ", " Rating ": " calcite_Rating ", " rating ": " calcite_rating ",
           " RANGE ": " calcite_RANGE ", " Range ": " calcite_Range ", " range ": " calcite_range ",
           " CHARACTER ": " calcite_CHARACTER ", " Character ": " calcite_Character ",
           " count ": " calcite_count "}
    # use these 3 lines to do the replacement
    rep = dict((re.escape(k), v) for k, v in rep.items())
    # Python 3 renamed dict.iteritems to dict.items so use rep.items() for latest versions
    pattern = re.compile("|".join(rep.keys()))
    textsql = pattern.sub(lambda m: rep[re.escape(m.group(0))], textsql)
    # special edits to fit for calcite
    # textsql = textsql.replace('SUBSTR', 'SUBSTRING').replace('STRFcalcite_TIME', 'STRFTIME').replace(
    #     'strfcalcite_time', 'strftime')
    return textsql


def get_promo_pools(promo_df):
    pools = {}
    db_id_pool_pos = []
    query_pool_pos = []
    # pos_rewrites = []
    rule_path_pool_pos = []
    db_id_pool_neg = []
    query_pool_neg = []
    rule_path_pool_neg = []
    for index, row in promo_df.iterrows():
        db_id = row['db_id']
        query = str(row['original_sql'])
        query = edit_queries(str(row['original_sql']))
        # query_r = str(row['rewritten_sql_gpt'])
        rule_path = row['activated_rules_gpt']
        if row['latency_org'] != 'NA' and row['latency_gpt'] != 'NA':
            prop = float(row['latency_gpt']) / float(row['latency_org'])
            if prop < 1 and rule_path != '[]':
                db_id_pool_pos.append(db_id)
                query_pool_pos.append(query)
                # pos_rewrites.append(query_r)
                rule_path_pool_pos.append(rule_path)
            elif prop > 1 and rule_path != '[]':
                db_id_pool_neg.append(db_id)
                query_pool_neg.append(query)
                rule_path_pool_neg.append(rule_path)
    pools['pos'] = (db_id_pool_pos, query_pool_pos, rule_path_pool_pos)
    pools['neg'] = (db_id_pool_neg, query_pool_neg, rule_path_pool_neg)
    # df_pos = {'db_id_pool_pos': db_id_pool_pos, 'query_pool_pos': query_pool_pos,
    #           'rewrite_pool_pos': pos_rewrites, 'rule_path_pool_pos': rule_path_pool_pos}
    # df_pos = pd.DataFrame(df_pos)
    # df_pos.to_csv('pos_pool.csv')
    return pools


def simple_distance(A, B):
    if A.label != B.label:
        return 1
    return 0


def list_to_zss_tree(lst):
    """
    Converts a nested list representation of a binary tree to a zss Node tree.
    The nested list should be in the format: [root, left_subtree, right_subtree]
    input lst: The nested list.
    output: The root node of the corresponding zss tree.
    """
    # Base case: if the list is empty, return None
    if not lst:
        return None
    # Create the root node
    if isinstance(lst, list):
        root = zss.Node(lst[0])
        # If there's a left subtree, recursively convert it and attach to root
        if len(lst) > 1 and lst[1]:
            root.addkid(list_to_zss_tree(lst[1]))
        # If there's a right subtree, recursively convert it and attach to root
        if len(lst) > 2 and lst[2]:
            root.addkid(list_to_zss_tree(lst[2]))
    else:
        root = zss.Node(lst)
    return root


def get_top_k_smallest_indices(in_list, k):
    # Get the indices sorted by the numbers
    sorted_list = sorted(in_list)
    out_inds = []
    for i in range(len(sorted_list)):
        small_val = sorted_list[i]
        if i <= k - 1:
            inds = [ind for ind in range(len(sorted_list)) if in_list[ind] == small_val and ind not in out_inds]
            selected_ind = random.choices(inds, k=1)[0]
            out_inds.append(selected_ind)
        else:
            break
    return out_inds


def get_k_promos(k, pos_pool, neg_pool, db_id, query, logical_plan, method='plan', same=False):
    db_id_pool_pos, query_pool_pos, rule_path_pool_pos, logical_plan_pos, embeddings_pos = pos_pool
    db_id_pool_neg, query_pool_neg, rule_path_pool_neg, logical_plan_neg, embeddings_neg = neg_pool
    db_id_pool_all = db_id_pool_pos + db_id_pool_neg
    query_pool_all = query_pool_pos + query_pool_neg
    rule_path_pool_all = rule_path_pool_pos + rule_path_pool_neg
    logical_plan_all = logical_plan_pos + logical_plan_neg
    if method == 'sentbert':
        embeddings_all = np.concatenate((embeddings_pos, embeddings_neg), axis=0)
    elif method == 'queryCL':
        enc2 = torch.cat((embeddings_pos, embeddings_neg))
    if query in query_pool_all and same:
        assert k == 1
        same_ind = query_pool_all.index(query)
        same_promos = [get_promo_meta(db_id_pool_all[same_ind], query_pool_all[same_ind], rule_path_pool_all[same_ind])]
        return same_promos
    if method == 'random':
        all_indices = np.arange(len(db_id_pool_all))
        rdm_inds = random.choices(all_indices, k=k)
        # print(rdm_inds)
        random_promos = []
        for i in rdm_inds:
            random_promos.append(get_promo_meta(db_id_pool_all[i], query_pool_all[i], rule_path_pool_all[i]))
        return random_promos

    elif method == 'plan':
        # print(logical_plan)
        if not logical_plan:
            print('invalid plan, use random')
            all_indices = np.arange(len(db_id_pool_all))
            rdm_inds = random.choices(all_indices, k=k)
            # print(rdm_inds)
            random_promos = []
            for i in rdm_inds:
                random_promos.append(get_promo_meta(db_id_pool_all[i], query_pool_all[i], rule_path_pool_all[i]))
            return random_promos
        else:
            tree_plan_test = list_to_zss_tree(logical_plan)
            # filter out itself
            tree_plans_pool = [list_to_zss_tree(ast.literal_eval(logical_plan_all[i])) if query != query_pool_all[i]
                               else None for i in range(len(logical_plan_all))]
            tree_edit_dists = [zss.simple_distance(tree_plan_test, tree) if tree else float('inf') for tree in
                               tree_plans_pool]
            sim_indices = get_top_k_smallest_indices(tree_edit_dists, k)
            print(sim_indices)
            plan_promos = []
            for i in sim_indices:
                print(str(logical_plan_all[i]) == str(logical_plan))
                plan_promos.append(get_promo_meta(db_id_pool_all[i], query_pool_all[i], rule_path_pool_all[i]))
            return plan_promos

    elif method == 'sentbert':
        sent_promos = []
        in_sent = [query]
        in_ind = -1
        # filter out itself
        if query in query_pool_all:
            in_ind = query_pool_all.index(query)
        # Sentences are encoded by calling model.encode()
        in_embedding = pre_lang_model.encode(in_sent)
        # print(in_embedding)
        sim_scores = []
        for i in range(len(embeddings_all)):
            if i != in_ind:
                sim_score = cosine_similarity(in_embedding, [embeddings_all[i]])[0][0]
                sim_scores.append(sim_score)
        sorted_scores = sorted(sim_scores, reverse=True)
        promo_inds = []
        for i in range(k):
            score = sorted_scores[i]
            promo_inds.append(sim_scores.index(score))
        for ind in promo_inds:
            sent_promos.append(get_promo_meta(db_id_pool_all[ind], query_pool_all[ind], rule_path_pool_all[ind]))
        return sent_promos

    elif method == 'queryCL':
        # filter out itself
        query_pool_all = [x for x in query_pool_all if x != query]
        batch1 = [[query]]
        enc1 = batcher(batch1, [db_id])
        enc1 = enc1.repeat((enc2.shape[0], 1))
        sim_scores = []
        for kk in range(enc2.shape[0]):
            sys_score = compute_similarity(enc1[kk], enc2[kk])
            sim_scores.append(sys_score)
        # all_sys_scores.extend(sys_scores)
        cl_promos = []
        sorted_scores = sorted(sim_scores, reverse=True)
        promo_inds = []
        for i in range(k):
            score = sorted_scores[i]
            promo_inds.append(sim_scores.index(score))
        for ind in promo_inds:
            cl_promos.append(get_promo_meta(db_id_pool_all[ind], query_pool_all[ind], rule_path_pool_all[ind]))
        return cl_promos


def append_logical_plans(in_csv):
    df_pool = pd.read_csv(in_csv)
    ids = df_pool['db_id'].tolist()
    originals = [edit_queries(x) for x in df_pool['original_sql'].tolist()]
    plans = []
    for i in range(len(ids)):
        plan = get_logical_plan(ids[i], originals[i])
        print(plan)
        plans.append(plan)
    df_pool['original_logical_plan'] = plans
    df_pool = pd.DataFrame(df_pool)
    df_pool.to_csv(in_csv)
    print('logical plan appended')


def get_pool(poll_csv, method):
    pool_df = pd.read_csv(poll_csv)
    sentences = [edit_queries(x) for x in pool_df['original_sql'].tolist()]
    if method == 'sentbert':
        embeddings = pre_lang_model.encode(sentences)
    elif method == 'queryCL':
        batch2 = [[x] for x in pool_df['original_sql'].tolist()]
        embeddings = batcher(batch2, pool_df['db_id'].tolist())
    else:
        embeddings = []
    promo_pool = (pool_df['db_id'].tolist(), pool_df['original_sql'].tolist(),
                  pool_df['activated_rules'].tolist(), pool_df['original_logical_plan'].tolist(), embeddings)
    return promo_pool


def LLM_R2(dataset, method, num_promos):
    demo_time_record = []
    llm_time_record = []
    rewriter_time_record = []
    df_gpt = {}
    db_ids = []
    original_queries = []
    rewritten_queries_s = []
    activated_rules_s = []
    prompt_queries_s = []
    prompt_rules_s = []
    # preprocess dataset, uncomment when running for the first time
    process_time_start = time.time()
    # append_logical_plans('pos_pool_' + dataset + '_updated.csv')
    # append_logical_plans('neg_pool_' + dataset + '_updated.csv')
    df_test = pd.read_csv('../data/data_llmr2/queries/queries_' + dataset + '_test.csv').fillna('NA')
    promo_pool_pos = get_pool('../data/data_llmr2/pools/pos_pool_' + dataset + '_updated.csv', method)
    promo_pool_neg = get_pool('../data/data_llmr2/pools/neg_pool_' + dataset + '_updated.csv', method)

    process_time_end = time.time()
    process_time = process_time_end - process_time_start
    print('preprocess time: ', process_time)
    print('query pool embeddings collected')
    for index, row in df_test.iterrows():
        if index >= 0:
            df_i = {}
            db_id = row['db_id']
            db_ids.append(db_id)
            query = row['original_sql'].replace(';', '') + ';'
            if query == 'NA;':
                original_queries.append('NA')
                rewritten_queries_s.append('NA')
                activated_rules_s.append('NA')
                prompt_queries_s.append('NA')
                prompt_rules_s.append('NA')
            else:
                with open('../data/data_llmr2/schemas/' + db_id + '.json') as f_sch:
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

                logical_plan = get_logical_plan(db_id, edit_queries(query))
                demo_time_start = time.time()
                sim_promos = get_k_promos(num_promos, promo_pool_pos, promo_pool_neg, db_id, query, logical_plan, method=method)
                demo_time_end = time.time()
                demo_time = demo_time_end - demo_time_start
                demo_time_record.append(demo_time)
                schema_0, query_0, logical_plan_0, rules_list_0 = sim_promos[0]

                sim_prompt = generate_turbo_prompt_light(schema, query, logical_plan, sim_promos)

                # attempt gpt api for max 3 times, if the previous try failed
                print(str(rules_list_0) != "['Empty List']")
                llm_time_start = time.time()
                if str(rules_list_0) != "['Empty List']":
                    trys = 0
                    gpt_output_s = query_gpt_attempts(sim_prompt, trys)
                    gpt_rules_s = filter_gpt_output(gpt_output_s)
                else:
                    gpt_rules_s = []
                llm_time_end = time.time()
                llm_time = llm_time_end - llm_time_start
                llm_time_record.append(llm_time)

                rewriter_time_start = time.time()
                rewrite_query_s = call_rewriter(db_id, query, gpt_rules_s)
                rewriter_time_end = time.time()
                rewriter_time = rewriter_time_end - rewriter_time_start
                rewriter_time_record.append(rewriter_time)

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

                rewritten_queries_s.append(rewrite_query_s)
                if gpt_rules_s:
                    activated_rules_s.append(gpt_rules_s)
                else:
                    activated_rules_s.append([])

                promo_queries = []
                promo_rules = []
                for i in range(len(sim_promos)):
                    _, query_0, _, rules_list_0 = sim_promos[i]
                    promo_queries.append(query_0)
                    promo_rules.append(rules_list_0)
                prompt_queries_s.append(promo_queries)
                prompt_rules_s.append(promo_rules)

                print(query)
                # print(rewrite_query)
                print(gpt_rules_s)

            if index % 500 == 0 and index > 0:
                print(index)
                df_i['db_id'] = db_ids
                df_i['original_sql'] = original_queries
                df_i['rewritten_sql_gpt'] = rewritten_queries_s
                df_i['activated_rules_gpt'] = activated_rules_s
                df_i['prompt_sql_similar'] = prompt_queries_s
                df_i['prompt_rules_similar'] = prompt_rules_s
                df_i = pd.DataFrame(df_i)
                df_i.to_csv('../results/gpt_' + dataset + '_one_promo_' + method + '_updated_to_' + str(index) + '.csv')

                df_t_i = {'demo_time': demo_time_record, 'llm_time': llm_time_record, 'rewriter_time': rewriter_time_record}
                df_t_i = pd.DataFrame(df_t_i)
                df_t_i.to_csv('../results/time_gpt_' + dataset + '_one_promo_' + method + '_cleaned_to_' + str(index) + '.csv')
                # break
        # break
    df_gpt['db_id'] = db_ids
    df_gpt['original_sql'] = original_queries
    df_gpt['rewritten_sql_gpt'] = rewritten_queries_s
    df_gpt['activated_rules_gpt'] = activated_rules_s
    df_gpt['prompt_sql_similar'] = prompt_queries_s
    df_gpt['prompt_rules_similar'] = prompt_rules_s
    df_gpt = pd.DataFrame(df_gpt)
    df_gpt.to_csv('../results/gpt_' + dataset + '_one_promo_' + method + '_updated.csv')

    df_t = {'demo_time': demo_time_record, 'llm_time': llm_time_record, 'rewriter_time': rewriter_time_record}
    df_t = pd.DataFrame(df_t)
    df_t.to_csv('../results/time_gpt_' + dataset + '_one_promo_' + method + '_cleaned.csv')


# df_test = pd.read_csv('data/queries/queries_tpch_test.csv').fillna('NA')
# method = 'sentbert'
# promo_pool_pos = get_pool('pos_pool_tpch_updated.csv', method)
# promo_pool_neg = get_pool('neg_pool_tpch_best.csv', method)
# df_test = pd.read_csv('data/queries/queries_job_syn_test.csv').fillna('NA')
# method = 'plan'
# promo_pool_pos = get_pool('pos_pool_job_syn.csv', method)
# promo_pool_neg = get_pool('neg_pool_job_syn.csv', method)
method = 'queryCL'
dataset = 'dsb'
num_promos = 1
LLM_R2(dataset, method, num_promos)
