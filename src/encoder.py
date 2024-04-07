# @TIME : 3/10/23 12:02 PM
# @AUTHOR : LZDH

import os
import math
import numpy as np
import torch
import torch.nn as nn
from collections import deque
from sentence_transformers import SentenceTransformer
from get_query_meta import *
import torch.nn.functional as F

os.environ["TOKENIZERS_PARALLELISM"] = "false"
model = SentenceTransformer('all-MiniLM-L6-v2')


rel_list = ['LogicalAgg', 'LogicalCalc', 'LogicalCorrelate', 'LogicalExchange', 'LogicalFilter', 'LogicalIntersect',
            'LogicalJoin', 'LogicalMatch', 'LogicalMinus', 'LogicalProject', 'LogicalRepeatUnion', 'LogicalSnapshot',
            'LogicalSort', 'LogicalSortExchange', 'LogicalTableFunctionScan', 'LogicalTableModify', 'LogicalTableScan',
            'LogicalTableSpool', 'LogicalUnion', 'LogicalValues', 'LogicalWindow']
# print(len(rel_list))


class TreeNode:
    def __init__(self, dictionary):
        self.level = 0  # root

        self.nodeType = None
        self.nodeInfo = None
        self.cost_est = None
        self.card_est = None
        self.rowcount_est = None

        self.children = []
        self.parent = None
        self.text = None

        self.__dict__.update(dictionary)

    def update(self, dictionary):
        self.__dict__.update(dictionary)

    def __str__(self):
        #        return TreeNode.print_nested(self)
        return '{} with {}, {}, {}, {}, {} children'.format(self.nodeType, self.nodeInfo, self.cost_est,
                                                            self.card_est, self.rowcount_est, len(self.children))

    def __repr__(self):
        return self.__str__()

    def print_nested(self):
        print('--' * self.level + self.__str__())
        for k in self.children:
            if k:
                k.print_nested()


def extractNode(node):
    d = {
        'nodeType': node['NodeType'],
        'nodeInfo': node['Node_Detail'],
        'card_est': math.log10(float(node['Cum_Rows'])),
        'cost_est': math.log10(float(node['Cum_Cpu'])),
        'rowcount_est': math.log10(float(node['Rows'])),
        'text': 'The node is ' + node['NodeType'] + ' with details ' + node['Node_Detail'] + '.',
    }
    return d


def traversePlan(root, level=0):
    if isinstance(root, list):
        root_dict = root[0]
        root_node = TreeNode(extractNode(root_dict))
        root_node.level = level
        if len(root) > 1:
            for child in root[1:]:
                node = traversePlan(child, level + 1)
                node.parent = root_node
                root_node.children.append(node)
    else:
        root_dict = root
        root_node = TreeNode(extractNode(root_dict))
        root_node.level = level
    return root_node


# node = traversePlan(get_physical_tree(db_id, sql_input))
# node_1 = traversePlan(get_physical_tree(db_id, sql_input_1))
# node_2 = traversePlan(get_physical_tree(db_id, sql_input_2))
# print(node)
# print(node.text)
# print(node.children[0])
# print(node.children[0].text)


def node2feature(node, model):
    nodetype = node.nodeType
    type_feature = [1 if x == nodetype else 0 for x in rel_list]
    detail_feature = model.encode([node.nodeInfo])[0]
    cost_feature = [node.card_est, node.cost_est, node.rowcount_est]
    features = np.concatenate((type_feature, detail_feature, cost_feature))
    # print(features)
    # feature = model.encode([node.text])[0]
    return features


def floyd_warshall_rewrite(adjacency_matrix):
    (nrows, ncols) = adjacency_matrix.shape
    assert nrows == ncols
    M = adjacency_matrix.copy().astype('long')
    for i in range(nrows):
        for j in range(ncols):
            if i == j:
                M[i][j] = 0
            elif M[i][j] == 0:
                # M[i][j] = 510
                M[i][j] = 60
    for k in range(nrows):
        for i in range(nrows):
            for j in range(nrows):
                M[i][j] = min(M[i][j], M[i][k] + M[k][j])
    return M


def pad_1d_unsqueeze(x, padlen):
    x = x + 1  # pad id = 0
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen], dtype=x.dtype)
        new_x[:xlen] = x
        x = new_x
    return x.unsqueeze(0)


def pad_2d_unsqueeze(x, padlen):
    # dont know why add 1, comment out first
    #    x = x + 1 # pad id = 0
    xlen, xdim = x.size()
    # xlen = len(x)
    # pad_x = [1, 1, 1, 1, '', 1]
    if xlen < padlen:
        #     i = xlen
        #     while i < padlen:
        #         x.append(pad_x)
        #         i += 1
        # return x
        new_x = x.new_zeros([padlen, xdim], dtype=x.dtype) + 1
        new_x[:xlen, :] = x
        x = new_x
    else:
        print('exceed max pad: ', xlen)
    return x.unsqueeze(0)


def pad_rel_pos_unsqueeze(x, padlen):
    x = x + 1
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen], dtype=x.dtype)
        new_x[:xlen, :xlen] = x
        x = new_x
    return x.unsqueeze(0)


def pad_attn_bias_unsqueeze(x, padlen):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen], dtype=x.dtype).fill_(float('-inf'))
        new_x[:xlen, :xlen] = x
        new_x[xlen:, :xlen] = 0
        x = new_x
    return x.unsqueeze(0)


class QueryFormerPreprocess:
    def __init__(self, input_node, model, max_node=200, rel_pos_max=20): # old 60
        self.model = model
        self.max_node = max_node
        self.rel_pos_max = rel_pos_max
        # self.nodes = nodes
        # self.collated_dicts = [self.pre_collate(self.node2dict(node)) for node in nodes]
        self.output = self.pre_collate(self.node2dict(input_node))

    def __getitem__(self):
        return self.output

    ## pre-process first half of old collator
    def pre_collate(self, the_dict):
        ## input is the 'dict'
        # x = the_dict['features']
        x = pad_2d_unsqueeze(the_dict['features'], self.max_node)
        N = len(the_dict['features'])
        attn_bias = torch.zeros([N + 1, N + 1], dtype=torch.float)

        edge_index = the_dict['adjacency_list'].t()
        if len(edge_index) == 0:
            shortest_path_result = np.array([[0]])
            path = np.array([[0]])
            adj = torch.tensor([[0]]).bool()
        else:
            adj = torch.zeros([N, N], dtype=torch.bool)
            adj[edge_index[0, :], edge_index[1, :]] = True
            shortest_path_result = floyd_warshall_rewrite(adj.numpy())

        rel_pos = torch.from_numpy((shortest_path_result)).long()
        attn_bias[1:, 1:][rel_pos >= self.rel_pos_max] = float('-inf')

        attn_bias = pad_attn_bias_unsqueeze(attn_bias, self.max_node + 1)
        rel_pos = pad_rel_pos_unsqueeze(rel_pos, self.max_node)
        heights = pad_1d_unsqueeze(the_dict['heights'], self.max_node)

        return {
            'x': x,
            'attn_bias': attn_bias,
            'rel_pos': rel_pos,
            'heights': heights
        }

    def node2dict(self, node):
        adj_list, num_child, features = self.topo_sort(node)
        heights = self.calculate_height(adj_list, len(features))
        return {
            'features': torch.FloatTensor(np.array(features)),
            'heights': torch.LongTensor(heights),
            'adjacency_list': torch.LongTensor(np.array(adj_list)),
        }

    def topo_sort(self, root_node):
        adj_list = []  # from parent to children
        num_child = []
        features = []
        toVisit = deque()
        toVisit.append((0, root_node))
        next_id = 1
        while toVisit:
            idx, node = toVisit.popleft()
            node_features = node2feature(node, self.model)
            features.append(node_features)
            num_child.append(len(node.children))
            for child in node.children:
                # child.query_id = node.query_id
                toVisit.append((next_id, child))
                adj_list.append((idx, next_id))
                next_id += 1
        return adj_list, num_child, features

    def calculate_height(self, adj_list, tree_size):
        if tree_size == 1:
            return np.array([0])
        adj_list = np.array(adj_list)
        node_ids = np.arange(tree_size, dtype=int)
        node_order = np.zeros(tree_size, dtype=int)
        uneval_nodes = np.ones(tree_size, dtype=bool)
        parent_nodes = adj_list[:, 0]
        child_nodes = adj_list[:, 1]
        n = 0
        while uneval_nodes.any():
            uneval_mask = uneval_nodes[child_nodes]
            unready_parents = parent_nodes[uneval_mask]

            node2eval = uneval_nodes & ~np.isin(node_ids, unready_parents)
            node_order[node2eval] = n
            uneval_nodes[node2eval] = False
            n += 1
        return node_order


def collator(node_feature_set):
    x_row = []
    attn_bias_row = []
    rel_pos_row = []
    heights_row = []
    for row in node_feature_set:
        x_row.append([s['x'] for s in row])
        attn_bias_row.append([s['attn_bias'] for s in row])
        rel_pos_row.append([s['rel_pos'] for s in row])
        heights_row.append([s['heights'] for s in row])
        # x_row.append(torch.cat([s['x'] for s in row]))
        # attn_bias_row.append(torch.cat([s['attn_bias'] for s in row]))
        # rel_pos_row.append(torch.cat([s['rel_pos'] for s in row]))
        # heights_row.append(torch.cat([s['heights'] for s in row]))
    x = x_row
    attn_bias = attn_bias_row
    rel_pos = rel_pos_row
    heights = heights_row
    # x = torch.cat(x_row)
    # attn_bias = torch.cat(attn_bias_row)
    # rel_pos = torch.cat(rel_pos_row)
    # heights = torch.cat(heights_row)
    # return Batch(attn_bias, rel_pos, heights, x)
    return {'attn_bias': attn_bias, 'rel_pos': rel_pos, 'heights': heights, 'x': x}


def eval_collator(node_feature_set):
    x_row = []
    attn_bias_row = []
    rel_pos_row = []
    heights_row = []
    for row in node_feature_set:
        x_row.append(torch.cat([s['x'] for s in row]))
        attn_bias_row.append(torch.cat([s['attn_bias'] for s in row]))
        rel_pos_row.append(torch.cat([s['rel_pos'] for s in row]))
        heights_row.append(torch.cat([s['heights'] for s in row]))
    x = torch.cat(x_row)
    attn_bias = torch.cat(attn_bias_row)
    rel_pos = torch.cat(rel_pos_row)
    heights = torch.cat(heights_row)
    # return Batch(attn_bias, rel_pos, heights, x)
    return {'attn_bias': attn_bias, 'rel_pos': rel_pos, 'heights': heights, 'x': x}


def prepare_enc_data(query_dataset, model, db_ids):
    # dim of dataset is bs x query pair (org, pos / org, pos, hard neg)
    # first assuming selecting from the same db_id
    processed_dataset = []
    for i in range(len(query_dataset)):
        db_id = db_ids[i]
        nodes = [traversePlan(get_physical_tree(db_id, sql_input)) for sql_input in query_dataset[i]]
        processed_dataset.append([QueryFormerPreprocess(node, model).__getitem__() for node in nodes])
    return processed_dataset


class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(hidden_size, ffn_size)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(ffn_size, hidden_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, attention_dropout_rate, head_size):
        super(MultiHeadAttention, self).__init__()

        self.head_size = head_size

        self.att_size = att_size = hidden_size // head_size
        self.scale = att_size ** -0.5

        self.linear_q = nn.Linear(hidden_size, head_size * att_size)
        self.linear_k = nn.Linear(hidden_size, head_size * att_size)
        self.linear_v = nn.Linear(hidden_size, head_size * att_size)
        self.att_dropout = nn.Dropout(attention_dropout_rate)

        self.output_layer = nn.Linear(head_size * att_size, hidden_size)

    def forward(self, q, k, v, attn_bias=None):
        orig_q_size = q.size()

        d_k = self.att_size
        d_v = self.att_size
        batch_size = q.size(0)

        # head_i = Attention(Q(W^Q)_i, K(W^K)_i, V(W^V)_i)
        q = self.linear_q(q).view(batch_size, -1, self.head_size, d_k)
        k = self.linear_k(k).view(batch_size, -1, self.head_size, d_k)
        v = self.linear_v(v).view(batch_size, -1, self.head_size, d_v)

        q = q.transpose(1, 2)  # [b, h, q_len, d_k]
        v = v.transpose(1, 2)  # [b, h, v_len, d_v]
        k = k.transpose(1, 2).transpose(2, 3)  # [b, h, d_k, k_len]

        # Scaled Dot-Product Attention.
        # Attention(Q, K, V) = softmax((QK^T)/sqrt(d_k))V
        q = q * self.scale
        x = torch.matmul(q, k)  # [b, h, q_len, k_len]
        if attn_bias is not None:
            x = x + attn_bias

        x = torch.softmax(x, dim=3)
        x = self.att_dropout(x)
        x = x.matmul(v)  # [b, h, q_len, attn]

        x = x.transpose(1, 2).contiguous()  # [b, q_len, h, attn]
        x = x.view(batch_size, -1, self.head_size * d_v)

        x = self.output_layer(x)

        assert x.size() == orig_q_size
        return x


class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate, attention_dropout_rate, head_size):
        super(EncoderLayer, self).__init__()

        self.self_attention_norm = nn.LayerNorm(hidden_size)
        self.self_attention = MultiHeadAttention(hidden_size, attention_dropout_rate, head_size)
        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)

    def forward(self, x, attn_bias=None):
        y = self.self_attention_norm(x)
        y = self.self_attention(y, y, y, attn_bias)
        y = self.self_attention_dropout(y)
        x = x + y

        y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y
        return x


class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp


class MLP_layer(nn.Module):
    def __init__(self, hidden_size, ffn_size, output_size, dropout_rate):
        super(MLP_layer, self).__init__()

        self.layer1 = nn.Linear(hidden_size, ffn_size)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(ffn_size, output_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        return x


class QueryFormerEncoder(nn.Module):
    def __init__(self, hidden_dim=21 + 384 + 3, ffn_dim=32, head_size=8, dropout=0.01,
                 attention_dropout_rate=0.01, n_layers=8, eval=False):
        # tpch: 21 + 384 + 3
        # job: 300 + 384 + 300, all use 128 hidden
        # dsb: 30 + 30 + 30
        super(QueryFormerEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.head_size = head_size
        self.rel_pos_encoder = nn.Embedding(64, head_size, padding_idx=0)
        self.height_encoder = nn.Embedding(64, hidden_dim, padding_idx=0)

        self.input_dropout = nn.Dropout(dropout)
        encoders = [EncoderLayer(hidden_dim, ffn_dim, dropout, attention_dropout_rate, head_size)
                    for _ in range(n_layers)]
        self.layers = nn.ModuleList(encoders)
        self.final_ln = nn.LayerNorm(hidden_dim)
        self.super_token = nn.Embedding(1, hidden_dim)
        self.super_token_virtual_distance = nn.Embedding(1, head_size)
        # self.ffn = FeedForwardNetwork(3, 64, 0)
        # self.mlp_o = MLP_layer(21, 128, 30, 0)
        self.mlp_c = MLP_layer(3, 128, 3, 0)
        # self.mlp_cond = MLP_layer(384, 128, 30, 0)
        # self.ffn_out = FeedForwardNetwork(hidden_dim, 5, 0)

        self.fc1 = nn.Linear(hidden_dim, 128)
        self.fc2 = nn.Linear(128, hidden_dim)
        self.activation_fn = nn.GELU()

        self.eval = eval

    # batch data is pre-processed by QueryFormerPreprocess
    def forward(self, attn_bias, rel_pos, x, heights):
        type_feature, detail_feature, cost_feature = torch.split(x, (21, 384, 3), dim=-1)


        # cost_feature = self.ffn(cost_feature)
        # print(x.get_device())
        # print(type_feature.get_device())
        # type_feature = self.mlp_o(type_feature)
        # detail_feature = self.mlp_cond(detail_feature)
        cost_feature = self.mlp_c(cost_feature)


        # print(type_feature)
        x = torch.cat((type_feature, detail_feature, cost_feature), dim=-1)
        # print(x)
        heights = heights
        n_batch, n_node = x.size()[:2]
        tree_attn_bias = attn_bias.clone()
        tree_attn_bias = tree_attn_bias.unsqueeze(1).repeat(1, self.head_size, 1, 1)
        # rel pos
        # [n_batch, n_node, n_node, n_head] -> [n_batch, n_head, n_node, n_node]
        rel_pos_bias = self.rel_pos_encoder(rel_pos).permute(0, 3, 1, 2)
        tree_attn_bias[:, :, 1:, 1:] = tree_attn_bias[:, :, 1:, 1:] + rel_pos_bias
        # reset rel pos here
        t = self.super_token_virtual_distance.weight.view(1, self.head_size, 1)
        tree_attn_bias[:, :, 1:, 0] = tree_attn_bias[:, :, 1:, 0] + t
        tree_attn_bias[:, :, 0, :] = tree_attn_bias[:, :, 0, :] + t

        node_feature = x
        node_feature = node_feature + self.height_encoder(heights)
        super_token_feature = self.super_token.weight.unsqueeze(0).repeat(n_batch, 1, 1)
        super_node_feature = torch.cat([super_token_feature, node_feature], dim=1)

        # transfomrer encoder
        output = self.input_dropout(super_node_feature)
        for enc_layer in self.layers:
            output = enc_layer(output, tree_attn_bias)
        # output = self.final_ln(output)
        # output = self.ffn_out(output)

        residual = output
        output = self.final_ln(output)
        output = self.activation_fn(self.fc1(output))
        output = F.dropout(output, p=0.01, training=self.training)
        output = self.fc2(output)
        output = F.dropout(output, p=0.01, training=self.training)
        output = residual + output

        if self.eval:
            return output[:, 0, :].view(-1, self.hidden_dim)
        else:
            return output[:, 0, :].view(-1, 3, self.hidden_dim)


# db_id = 'tpch'
# sql_input = "select ps_partkey, sum(ps_supplycost * ps_availqty) as calcite_value from partsupp, supplier, nation where ps_suppkey = s_suppkey and s_nationkey = n_nationkey and n_name = 'PERU' group by ps_partkey having sum(ps_supplycost * ps_availqty) > ( select sum(ps_supplycost * ps_availqty) * 0.0001000000 from partsupp, supplier, nation where ps_suppkey = s_suppkey and s_nationkey = n_nationkey and n_name = 'PERU' ) order by calcite_value desc;"
# sql_input_1 = "select ps_partkey, sum(ps_supplycost * ps_availqty) as calcite_value from partsupp, supplier, nation where ps_suppkey = s_suppkey and s_nationkey = n_nationkey and n_name = 'AKSJ' group by ps_partkey having sum(ps_supplycost * ps_availqty) > ( select sum(ps_supplycost * ps_availqty) * 0.01000000 from partsupp, supplier, nation where ps_suppkey = s_suppkey and s_nationkey = n_nationkey and n_name = 'AKSJ' ) order by calcite_value desc;"
# sql_input_2 = "SELECT nation.n_name AS supp_nation, nation0.n_name AS cust_nation, EXTRACT(YEAR FROM t.l_shipdate) AS l_year, SUM(t.l_extendedprice * (1 - t.l_discount)) AS revenue FROM supplier INNER JOIN (SELECT * FROM lineitem WHERE l_shipdate >= DATE '1995-01-01' AND l_shipdate <= DATE '1996-12-31') AS t ON supplier.s_suppkey = t.l_suppkey INNER JOIN orders ON t.l_orderkey = orders.o_orderkey INNER JOIN customer ON orders.o_custkey = customer.c_custkey INNER JOIN nation ON supplier.s_nationkey = nation.n_nationkey INNER JOIN nation AS nation0 ON customer.c_nationkey = nation0.n_nationkey AND (nation.n_name = 'PERU' AND nation0.n_name = 'ROMANIA' OR nation.n_name = 'ROMANIA' AND nation0.n_name = 'PERU') GROUP BY nation.n_name, nation0.n_name, EXTRACT(YEAR FROM t.l_shipdate) ORDER BY nation.n_name, nation0.n_name, EXTRACT(YEAR FROM t.l_shipdate);"
# sql_input_3 = "select cntrycode, count(*) as numcust, sum(c_acctbal) as totacctbal from ( select substring(c_phone from 1 for 2) as cntrycode, c_acctbal from customer where substring(c_phone from 1 for 2) in ('22', '33', '23', '37', '34', '43', '36') and c_acctbal > ( select avg(c_acctbal) from customer where c_acctbal > 0.00 and substring(c_phone from 1 for 2) in ('22', '33', '23', '37', '34', '43', '36') ) and not exists ( select * from orders where o_custkey = c_custkey ) ) as custsale group by cntrycode order by cntrycode;"
# # test_dat = [[sql_input, sql_input_1, sql_input_2], [sql_input_1, sql_input, sql_input_2], [sql_input_2, sql_input, sql_input_1]]
# test_dat = [[sql_input, sql_input_1, sql_input_2],
#             [sql_input_1, sql_input_2, sql_input],
#             [sql_input_2, sql_input_3, sql_input_1],
#             [sql_input_3, sql_input_1, sql_input]]
# db_ids = ['tpch', 'tpch', 'tpch', 'tpch']
# test_dat_features = prepare_enc_data(test_dat, model, db_ids)
# batched_data = eval_collator(test_dat_features)
# # print(batched_data.x.size())
#
# encoder = QueryFormerEncoder()
# emb = encoder(batched_data['attn_bias'], batched_data['rel_pos'], batched_data['x'], batched_data['heights'])
#
# print(emb)
# print(emb.size())

# # columns
# z_1 = emb[:, 0, :]
# z_2 = emb[:, 1, :]
# z_3 = emb[:, 2, :]
# print(z_1.size())
# #
# sim = Similarity(1)
# print(sim(z_1, z_2))
# print(sim(z_1, z_3))



# padding one hot for node type?

# sent1 = '(sort0=[$1], dir0=[DESC]).'
# sent2 = '(sort0=[$2], dir0=[DESC]).'
# sent3 = '(sort0=[$1], dir0=[ASCE]).'
# sent4 = '(abaaba=[$1], dir0=[DESC]).'
# num1 = '41696943328442.79'
# num2 = '33750192876391'
# num3 = '337501928763917234.0'
#
# f1 = torch.Tensor(model.encode([sent1]))
# f2 = torch.Tensor(model.encode([sent2]))
# f3 = torch.Tensor(model.encode([sent3]))
# f4 = torch.Tensor(model.encode([sent4]))
# n1 = torch.Tensor(model.encode([num1]))
# n2 = torch.Tensor(model.encode([num2]))
# n3 = torch.Tensor(model.encode([num3]))
# sim = Similarity(1)
# print(sim(f1, f2))
# print(sim(f1, f3))
# print(sim(f1, f4))
# print(sim(f3, f4))
#
# print(sim(n1, n2))
# print(sim(n1, n3))


