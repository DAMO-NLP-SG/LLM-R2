import torch
import torch.nn as nn
import torch.distributed as dist

import transformers
from transformers.activations import gelu
from transformers.modeling_outputs import SequenceClassifierOutput
from encoder import *

from sentence_transformers import SentenceTransformer


class MLPLayer(nn.Module):
    """
    Head for getting sentence representations over RoBERTa/BERT's CLS representation.
    """

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = self.activation(x)

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


class Pooler(nn.Module):
    """
    Parameter-free poolers to get the sentence embedding
    'cls': [CLS] representation with BERT/RoBERTa's MLP pooler.
    'cls_before_pooler': [CLS] representation without the original MLP pooler.
    'avg': average of the last layers' hidden states at each token.
    'avg_top2': average of the last two layers.
    'avg_first_last': average of the first and the last layers.
    """
    def __init__(self):
        super().__init__()

    def forward(self, outputs):
        return outputs


def cl_init(cls):
    """
    Contrastive learning class init function.
    """
    # if cls.model_args.pooler_type == "cls":
    #     cls.mlp = MLPLayer(config)
    # cls.sim = Similarity(temp=cls.model_args.temp)
    cls.sim = Similarity(temp=0.05)
    cls.pooler = Pooler()
    # cls.init_weights()
    cls.device = torch.device('cpu')


def cl_forward(cls, encoder, is_fix=False, x=None, heights=None, attn_bias=None, rel_pos=None):
    outputs = encoder(x=x, heights=heights, attn_bias=attn_bias, rel_pos=rel_pos)
    # Pooling
    pooler_output = cls.pooler(outputs)
    return pooler_output, outputs


####lz changed here added both fixed pretrain and new model
def cl_vat_forward(cls, encoder, encoder_fixed, x=None, heights=None, attn_bias=None, rel_pos=None):

    pooler_output, outputs_rep = cl_forward(cls, encoder, False, x=x, heights=heights, attn_bias=attn_bias, rel_pos=rel_pos)
    # outputs_rep_fixed, _ = cl_forward(cls, encoder_fixed, True, input_ids, attention_mask,
    #                                                    token_type_ids, position_ids,
    #                                                    head_mask, inputs_embeds, output_attentions, return_dict,
    #                                                    mlm_input_ids)

    # Separate representation
    print(pooler_output.size(), outputs_rep.size()) # (bs, num_sent, size_hidden)
    z1, z2, z_negative = outputs_rep[:, 0], outputs_rep[:, 1], outputs_rep[:, 2]
    # z1_fix, z2_fix, z_negative_fix = outputs_rep_fixed[:,0], outputs_rep_fixed[:,1], outputs_rep_fixed[:,2]

    # # Gather all embeddings if using distributed training
    # if dist.is_initialized() and cls.training:
    #     # print("yes")
    #     # Dummy vectors for allgather
    #     z1_list = [torch.zeros_like(z1) for _ in range(dist.get_world_size())]
    #     z2_list = [torch.zeros_like(z2) for _ in range(dist.get_world_size())]
    #     z_negative_list = [torch.zeros_like(z2) for _ in range(dist.get_world_size())]
    #     # Allgather
    #     dist.all_gather(tensor_list=z1_list, tensor=z1.contiguous())
    #     dist.all_gather(tensor_list=z2_list, tensor=z2.contiguous())
    #     dist.all_gather(tensor_list=z_negative_list, tensor=z_negative.contiguous())
    #     # Since allgather results do not have gradients, we replace the
    #     # current process's corresponding embeddings with original tensors
    #     z1_list[dist.get_rank()] = z1
    #     z2_list[dist.get_rank()] = z2
    #     z_negative_list[dist.get_rank()] = z_negative
    #     # Get full batch embeddings: (bs x N, hidden)
    #     z1 = torch.cat(z1_list, 0)
    #     z2 = torch.cat(z2_list, 0)
    #     z_negative = torch.cat(z_negative_list, 0)
    #
    #     # z1_list_fix = [torch.zeros_like(z1_fix) for _ in range(dist.get_world_size())]
    #     # z2_list_fix = [torch.zeros_like(z2_fix) for _ in range(dist.get_world_size())]
    #     # z_negative_list_fix = [torch.zeros_like(z_negative_fix) for _ in range(dist.get_world_size())]
    #     # # Allgather
    #     # dist.all_gather(tensor_list=z1_list_fix, tensor=z1_fix.contiguous())
    #     # dist.all_gather(tensor_list=z2_list_fix, tensor=z2_fix.contiguous())
    #     # dist.all_gather(tensor_list=z_negative_list_fix, tensor=z_negative_fix.contiguous())
    #     # # Since allgather results do not have gradients, we replace the
    #     # # current process's corresponding embeddings with original tensors
    #     # z1_list_fix[dist.get_rank()] = z1_fix
    #     # z2_list_fix[dist.get_rank()] = z2_fix
    #     # z_negative_list_fix[dist.get_rank()] = z_negative_fix
    #     # # Get full batch embeddings: (bs x N, hidden)
    #     # z1_fix = torch.cat(z1_list_fix, 0)
    #     # z2_fix = torch.cat(z2_list_fix, 0)
    #     # z_negative_fix = torch.cat(z_negative_list_fix, 0)

    batch_size, hidden_size = z2.size()
    cos_sim = cls.sim(z1.unsqueeze(1), z2.unsqueeze(0))
    print(cos_sim.size())  # bs*bs
    # cos_sim_fix = cls.sim(z1_fix.unsqueeze(1), z2_fix.unsqueeze(0))

    labels = torch.arange(cos_sim.size(0)).long().to(cls.device)  # [0, 1, 2, ..., 127]
    # loss_fct = nn.CrossEntropyLoss()
    cos_sim_negative = cls.sim(z1.unsqueeze(1), z_negative.unsqueeze(0))
    cos_sim = torch.cat([cos_sim, cos_sim_negative], 1)
    ####lz changed here, here cos_sim_negative and cos_sim_negative_fix are both from data instead of defined
    # cos_sim_negative_fix = cls.sim(z1_fix.unsqueeze(1), z_negative_fix.unsqueeze(0))
    # # Use all noise vectors as negatives
    # cos_sim_fix = torch.cat([cos_sim_fix, cos_sim_negative_fix], 1)

    labels_dis = torch.cat(
        [torch.eye(cos_sim.size(0), device=cos_sim.device)[labels], torch.zeros_like(cos_sim_negative)], -1)
    # todo: uncomment this filter=1 if need to filt in future
    # cls.model_args.phi = 1
    # weights = torch.where(cos_sim_fix > cls.model_args.phi * 20, 0, 1)
    # mask_weights = torch.eye(cos_sim.size(0), device=cos_sim.device) - torch.diag_embed(torch.diag(weights))
    # weights = weights + torch.cat([mask_weights, torch.zeros_like(cos_sim_negative)], -1)
    # soft_cos_sim = torch.softmax(cos_sim * weights, -1)

    soft_cos_sim = torch.softmax(cos_sim, -1)
    loss = - (labels_dis * torch.log(soft_cos_sim) + (1 - labels_dis) * torch.log(1 - soft_cos_sim))
    loss = torch.mean(loss)

    # loss = torch.mean(loss_fct(torch.log_softmax(cos_sim, -1), labels_dis) * weights)    #+ 0.1*kl_loss

    return SequenceClassifierOutput(
        loss=loss,
        logits=cos_sim,
        hidden_states=pooler_output,
    )


def sentemb_forward(cls, encoder, x=None, heights=None, attn_bias=None, rel_pos=None):

    outputs = encoder(x=x, heights=heights, attn_bias=attn_bias, rel_pos=rel_pos)

    return outputs


class QueryformerForCL(nn.Module):
    def __init__(self):
        super(QueryformerForCL, self).__init__()
        # self.model_args = model_kargs["model_args"]
        self.enc_tr = QueryFormerEncoder()
        self.enc_eval = QueryFormerEncoder(eval=True)
        self.fix_enc = QueryFormerEncoder()

        # fix_config = BertConfig.from_pretrained(self.model_args.c_model_name_or_path)
        # self.fix_bert = BertModel(fix_config)
        cl_init(self)

    def forward(self, x=None, heights=None, attn_bias=None, rel_pos=None, eval=None):
        if eval:
            return sentemb_forward(self, self.enc_eval, x=x, heights=heights, attn_bias=attn_bias,
                                  rel_pos=rel_pos)
        else:
            return cl_vat_forward(self, self.enc_tr, self.fix_enc, x=x, heights=heights, attn_bias=attn_bias,
                                  rel_pos=rel_pos)


# db_id = 'tpch'
# sql_input = "select ps_partkey, sum(ps_supplycost * ps_availqty) as calcite_value from partsupp, supplier, nation where ps_suppkey = s_suppkey and s_nationkey = n_nationkey and n_name = 'PERU' group by ps_partkey having sum(ps_supplycost * ps_availqty) > ( select sum(ps_supplycost * ps_availqty) * 0.0001000000 from partsupp, supplier, nation where ps_suppkey = s_suppkey and s_nationkey = n_nationkey and n_name = 'PERU' ) order by calcite_value desc;"
# sql_input_1 = "select ps_partkey, sum(ps_supplycost * ps_availqty) as calcite_value from partsupp, supplier, nation where ps_suppkey = s_suppkey and s_nationkey = n_nationkey and n_name = 'AKSJ' group by ps_partkey having sum(ps_supplycost * ps_availqty) > ( select sum(ps_supplycost * ps_availqty) * 0.01000000 from partsupp, supplier, nation where ps_suppkey = s_suppkey and s_nationkey = n_nationkey and n_name = 'AKSJ' ) order by calcite_value desc;"
# sql_input_2 = "SELECT nation.n_name AS supp_nation, nation0.n_name AS cust_nation, EXTRACT(YEAR FROM t.l_shipdate) AS l_year, SUM(t.l_extendedprice * (1 - t.l_discount)) AS revenue FROM supplier INNER JOIN (SELECT * FROM lineitem WHERE l_shipdate >= DATE '1995-01-01' AND l_shipdate <= DATE '1996-12-31') AS t ON supplier.s_suppkey = t.l_suppkey INNER JOIN orders ON t.l_orderkey = orders.o_orderkey INNER JOIN customer ON orders.o_custkey = customer.c_custkey INNER JOIN nation ON supplier.s_nationkey = nation.n_nationkey INNER JOIN nation AS nation0 ON customer.c_nationkey = nation0.n_nationkey AND (nation.n_name = 'PERU' AND nation0.n_name = 'ROMANIA' OR nation.n_name = 'ROMANIA' AND nation0.n_name = 'PERU') GROUP BY nation.n_name, nation0.n_name, EXTRACT(YEAR FROM t.l_shipdate) ORDER BY nation.n_name, nation0.n_name, EXTRACT(YEAR FROM t.l_shipdate)"
#
# pre_lang_model = SentenceTransformer('all-MiniLM-L6-v2')
#
# test_dat = [[sql_input, sql_input_1, sql_input_2],
#             [sql_input_1, sql_input, sql_input_2],
#             [sql_input_2, sql_input, sql_input_1]]
# db_ids = ['tpch', 'tpch', 'tpch']
# test_dat_features = prepare_enc_data(test_dat, pre_lang_model, db_ids)
# batched_data = collator(test_dat_features)
# print(batched_data.x.size())
#
# model00 = QueryformerForCL(pre_lang_model)
# result = model00(batched_data)
# print(result)
