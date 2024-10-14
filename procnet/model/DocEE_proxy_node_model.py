import torch
from torch import nn
import torch.nn.functional as F
from procnet.model.basic_model import BasicModel
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
from procnet.data_preparer.basic_preparer import BasicPreparer
from procnet.data_preparer.DocEE_preparer import DocEEPreparer
from typing import Dict
from transformers import PreTrainedModel
from torch_geometric.nn import FiLMConv
from procnet.conf.DocEE_conf import DocEEConfig
from procnet.utils.util_structure import UtilStructure


class DocEEBasicModel(BasicModel):
    def __init__(self,
                 preparer: BasicPreparer,):
        super().__init__()
        self.preparer = preparer
        self.b_tag_index, self.i_tag_index, self.one_b_tag_index, self.one_i_tag_index = self.init_bio_tag_index_information()
        self.language_model: PreTrainedModel = None

    def init_bio_tag_index_information(self) -> (set, set, int, int):
        b_tag = []  
        i_tag = []
        for tag in self.preparer.seq_BIO_index_to_tag:
            if tag.endswith('-B'):
                b_tag.append(tag)
            elif tag.endswith('-I'):
                i_tag.append(tag)
        the_chosen_tag = b_tag[0][:-2] #AveragePrice
        the_chosen_b_tag_index = self.preparer.seq_BIO_tag_to_index[the_chosen_tag + '-B'] #1 b中第一个index
        the_chosen_i_tag_index = self.preparer.seq_BIO_tag_to_index[the_chosen_tag + '-I'] #25 i中第一个index
        b_tag_index = [self.preparer.seq_BIO_tag_to_index[x] for x in b_tag] #[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
        i_tag_index = [self.preparer.seq_BIO_tag_to_index[x] for x in i_tag] #[26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44] i标签对应的index
        return set(b_tag_index), set(i_tag_index), the_chosen_b_tag_index, the_chosen_i_tag_index

    def get_bio_positions(self, bio_res: list, input_prob: bool, binary_mode: bool = False, input_id_int=None, ignore_padding_token: bool = True) -> list:
        if binary_mode:
            if input_prob:
                for x in bio_res: #bio_res[句子tokens数,49]  
                    sum_b = 0
                    for i in self.b_tag_index:
                        sum_b += x[i]
                    sum_i = 0
                    for i in self.i_tag_index:
                        sum_i += x[i]
                    x[self.one_b_tag_index] = sum_b #将所有b的全部放到oneb
                    x[self.one_i_tag_index] = sum_i #将所有i的全部放到onei  修改了原本bio_res 1，25的位置
        if input_prob:
            bio_index = []
            for x in bio_res:
                index = UtilStructure.find_max_number_index(x) #找到最大,也就是一个seq的每个token最有可能是b还是i还是o
                bio_index.append(index)
        else:
            bio_index = bio_res

        if ignore_padding_token:
            final_index = []
            before_has_pad = False
            for i in range(len(bio_index)):
                if input_id_int is not None:
                    if input_id_int[i] != self.preparer.get_auto_tokenizer().pad_token_id:
                        final_index.append(bio_index[i])
                        if before_has_pad:
                            raise Exception('get_bio_positions has padding token before other token!')
                    else:
                        before_has_pad = True
                else:
                    if bio_index[i] != -100:
                        final_index.append(bio_index[i])
                        if before_has_pad:
                            raise Exception('get_bio_positions has padding token before other token!')
                    else:
                        before_has_pad = True
        else: #不忽略
            final_index = [] #所有input_id_int的b=1 i=25 o=0的索引 
            for i in range(len(bio_index)): #句子token数
                if input_id_int is not None:
                    if input_id_int[i] != self.preparer.get_auto_tokenizer().pad_token_id: #！=0
                        final_index.append(bio_index[i])
                    else:
                        final_index.append(self.preparer.seq_BIO_tag_to_index['O'])
                else:
                    if bio_index[i] != -100:
                        final_index.append(bio_index[i])
                    else:
                        final_index.append(self.preparer.seq_BIO_tag_to_index['O'])

        if binary_mode: #这里为什么要重复做，final_index没有任何变化 二分也就是只有 b i o不细分 xxx-i xxx-b
            for i in range(len(final_index)):
                if final_index[i] in self.i_tag_index:
                    final_index[i] = self.one_i_tag_index
                elif final_index[i] in self.b_tag_index:
                    final_index[i] = self.one_b_tag_index

        bio_tag = [self.preparer.seq_BIO_index_to_tag[x] for x in final_index] #如果是binary_mode就是会对应到'AveragePrice-B' 'AveragePrice-I'
        # for the CLS of the first token, which should be 'O'
        bio_tag[0] = 'O'
        if bio_tag[1][-1] == 'I': #应该是修正，防止第一个是I，则修正为B
            bio_tag[1] = bio_tag[1][:-1] + 'B'
        position = BasicModel.find_BIO_spans_positions(bio_tag)
        bio_tag = self.validify_BIO_span(bio_tag, position, 'ignore')
        position = BasicModel.find_BIO_spans_positions(bio_tag)
        position = [tuple(pos) for pos in position] #spande index跨度
        return position


class DocEEGNNModelHN(nn.Module):
    def __init__(self, node_size, num_relations, dropout_ratio):
        super().__init__()
        self.node_size = node_size
        self.dropout_ratio = dropout_ratio
        self.gcn1 = FiLMConv(in_channels=node_size,
                             out_channels=node_size,
                             num_relations=num_relations,
                             )
        self.linear1 = nn.Sequential(
            nn.GELU(),
            nn.Linear(self.node_size, self.node_size),
            nn.Dropout(self.dropout_ratio),
        )

    def forward(self, x, edge_index, edge_type):
        x = self.gcn1(x=x, edge_index=edge_index, edge_type=edge_type)
        x = self.linear1(x)
        return x


class DocEEProxyNodeModel(DocEEBasicModel):
    def __init__(self,
                 config: DocEEConfig,
                 preparer: DocEEPreparer,
                 ):
        super().__init__(preparer=preparer)
        self.config = config
        self.slow_para = ['language_model']
        edge_types = ["M-self", "M-M", "C-M", "S-M"]

        self.dropout_ratio = 0.15
        self.node_size = config.node_size
        self.null_bio_index = preparer.seq_BIO_tag_to_index['O']
        assert self.null_bio_index == 0
        self.null_event_type_index = preparer.event_type_type_to_index['Null']
        self.null_event_relation_index = preparer.event_role_relation_to_index['Null']
        self.seq_BIO_index_to_tag = preparer.seq_BIO_index_to_tag
        self.event_type_index_to_type = preparer.event_type_index_to_type
        self.event_type_index_to_type_no_null = [x for x in self.event_type_index_to_type if x != 'Null']
        self.seq_bio_index_to_cate_no_null = preparer.seq_bio_index_to_cate[1:]
        self.num_BIO_tags = len(preparer.seq_BIO_index_to_tag)
        self.pos_event_ratio_total = preparer.pos_event_ratio_total
        self.neg_event_ratio_total = preparer.neg_event_ratio_total
        self.pos_bio_ratio_total = preparer.pos_bio_ratio_total
        self.neg_bio_ratio_total = preparer.neg_bio_ratio_total

        self.preparer = preparer
        self.num_proxy_slot = config.proxy_slot_num
        self.num_BIO_tag = len(preparer.seq_BIO_index_to_tag)
        self.num_event_type = len(preparer.event_type_type_to_index)
        self.num_event_relation = len(preparer.event_role_relation_to_index)

        self.mid_BIO_tag = self.num_BIO_tag // 2 + 1
        for i in range(self.mid_BIO_tag - 1):
            assert self.seq_BIO_index_to_tag[1 + i][:-2] == self.seq_BIO_index_to_tag[self.mid_BIO_tag + i][:-2] # B-XXX == I-XXX
            assert self.seq_bio_index_to_cate_no_null[i] == self.seq_BIO_index_to_tag[1 + i][:-2] # B-XXX == XXX

        self.language_model = self.new_bert_model(model_name=config.model_name)
        self.lm_size = self.language_model.config.hidden_size

        # (proxy_slot_num, proxy_size) 16，512 创建一个形状为 (self.num_proxy_slot, self.node_size) 的张量。
        self.initial_proxy_all_embedding = nn.Parameter(nn.init.xavier_uniform_(torch.FloatTensor(self.num_proxy_slot, self.node_size)))
        self.lm_bio_linear = nn.Sequential(
            nn.Dropout(self.dropout_ratio),
            nn.Linear(self.lm_size, self.lm_size // 4),
            nn.LayerNorm(self.lm_size // 4),
            nn.GELU(),
            nn.Dropout(self.dropout_ratio),
            nn.Linear(self.lm_size // 4, self.num_BIO_tag),
        ) #通过dropout、线性层、LayerNorm、GELU激活和再次dropout，最终将输入维度从lm_size降至num_BIO_tag，用于BIO标签分类。
        self.lm_hidden_linear = nn.Sequential(
            nn.Linear(self.lm_size + 1, self.node_size), #768+1，512
            nn.LayerNorm(self.node_size),
            nn.GELU(),
            nn.Dropout(self.dropout_ratio),
        )#将lm_size + 1维输入映射至node_size，经过LayerNorm和GELU激活后，进行dropout处理，用于隐藏层特征转换。
        '''
        用于处理语言模型（LM）的隐藏层特征。功能包括：
        将输入向量（维度为lm_size + 1）线性变换到node_size维度。
        对变换后的向量应用层规范化（LayerNorm）。
        使用GELU激活函数。
        以dropout_ratio概率对输出进行Dropout操作，防止过拟合。
        '''
        self.lm_cls_hidden_linear = nn.Sequential(
            nn.Linear(self.lm_size + 1, self.node_size), #768+1，512
            nn.LayerNorm(self.node_size),
            nn.GELU(),
            nn.Dropout(self.dropout_ratio),
        )
        self.edge_type_table = {edge_types[i]: i for i in range(len(edge_types))}
        self.gcn = DocEEGNNModelHN(node_size=self.node_size, num_relations=len(self.edge_type_table), dropout_ratio=self.dropout_ratio)

        self.proxy_slot_event_type_linear = nn.Sequential(
            nn.Linear(self.node_size, self.num_event_type),
        ) # 将nodesize映射到事件类型6
        self.proxy_span_attention = nn.MultiheadAttention(embed_dim=self.node_size, num_heads=8, dropout=self.dropout_ratio, batch_first=True)
        self.span_proxy_slot_relation_linear = nn.Sequential(
            nn.Linear(self.node_size + self.node_size, self.node_size // 4),
            nn.LayerNorm(self.node_size // 4),
            nn.GELU(),
            nn.Dropout(self.dropout_ratio),
            nn.Linear(self.node_size // 4, self.num_event_relation)
        )
        self.cls_total_event_num_linear = nn.Sequential(
            nn.Dropout(self.dropout_ratio),
            nn.Linear(self.node_size, self.node_size // 4),
            nn.LayerNorm(self.node_size // 4),
            nn.GELU(),
            nn.Dropout(self.dropout_ratio),
            nn.Linear(self.node_size // 4, 1),
        ) #将输入特征的维度降低到原来的四分之一，经过一个线性层输出单个值。 总事件数量
        self.span_span_relation_linear = nn.Sequential(
            nn.Dropout(self.dropout_ratio),
            nn.Linear(self.node_size * 2, 2),
        ) #两span间关系预测，判断实体是否属于同一个事件 
        self.ce_none_reduction_loss_fn = nn.CrossEntropyLoss(reduction='none') #交叉熵损失函数计算
        self.ce_normal_loss_fn = nn.CrossEntropyLoss()
        self.mse_loss_fn = nn.MSELoss()

    def forward(self,
                inputs_ids,
                inputs_att_masks,
                events_labels=None,
                bios_ids=None,
                use_mix_bio: bool = True,
                use_span_as_key: bool = True,
                ):
        # --- setup ---
        device = next(self.parameters()).device
        sentence_num = len(inputs_ids) #2

        # cpu [ [101, 102, 103], [102, 104, 105] ]
        input_ids_int = [x.detach().cpu().numpy().tolist() for x in inputs_ids]

        # --- entity record init ---
        bio_probs = []
        lm_clss_times = [] #  分类任务: 在后续阶段使用 lm_clss 进行分类决策。
        lm_hidden_state_times = [] #时序分析: 使用 lm_hidden_state 进行进一步的时序数据分析或输入到其他模型中
        position_times = [] #用于 记录每个时间步的实体位置信息，用于后续的事件关系预测。
        loss_bio = torch.FloatTensor([0]).to(device)
        # --- sequence labeling --- 进行命名实体识别（NER），使用 BERT 模型的最后一个隐层状态来预测 BERT 特征对应的标签（BIO）。
        for time_step in range(sentence_num):
            # (1, seq_length, )
            input_ids = inputs_ids[time_step].unsqueeze(0)
            # (1, seq_length, )
            bio_ids = bios_ids[time_step].unsqueeze(0) if bios_ids is not None else None
            # cpu [101, 102, 103]
            input_id_int = input_ids_int[time_step]
            # --- all sentence to LM for BIO ---
            lm_res: BaseModelOutputWithPoolingAndCrossAttentions = self.language_model(input_ids=input_ids) #bert做embedding
            # (1, seq_length, lm_size)
            lm_last_hidden_states = lm_res.last_hidden_state #inputs经过bert嵌入后的最后一个hidden_state
            # (1, seq_length, bio_tags_size)
            lm_logit = self.lm_bio_linear(lm_last_hidden_states) #通过一个线性层（lm_bio_linear）将最后的隐层状态转换为 BIO 标签的 logit（未归一化的预测值）。
            if bio_ids is None: # 判断是否进行评估
                one_loss_bio = torch.FloatTensor([0]).to(device) #如果是，表示当前是在评估（evaluation）阶段，这时损失（loss）设为 0。
            else: #计算损失。使用交叉熵损失函数 (ce_none_reduction_loss_fn) 来计算原始的损失值 raw_loss_bio。
                raw_loss_bio = self.ce_none_reduction_loss_fn(lm_logit.view(-1, self.num_BIO_tag), bio_ids.view(-1, )) #计算原始损失值 raw_loss_bio。
                bio_is_o = bio_ids.squeeze(0) == self.null_bio_index #找到标签为O（非实体）和非O的位置。
                bio_not_o = bio_ids.squeeze(0) != self.null_bio_index
                loss_o = torch.sum(raw_loss_bio * bio_is_o) #分别计算O标签和非O标签的损失值 loss_o 和 loss_bi
                loss_bi = torch.sum(raw_loss_bio * bio_not_o)
                one_loss_bio = loss_o * self.pos_bio_ratio_total + loss_bi * self.neg_bio_ratio_total #结合正负样本比例调整损失值，并乘以0.01进行缩放。 帮助模型更好地学习那些稀有但重要的样本（比如负样本）。
                one_loss_bio = one_loss_bio * 0.01
            loss_bio += one_loss_bio

            # (1, seq_length, bio_tags_size) 1,425,49   预测的bio
            bio_prob = F.softmax(lm_logit, dim=2) #得到bio概率分布
            # cpu probability of bio.  [[[0.1, 0.9], [0.5, 0.4 ]], ]
            bio_result = bio_prob.squeeze(0).detach().cpu().numpy().tolist()
            # cpu positions. [[[start, end], [start, end]], ]
            bio_probs.append(bio_prob.squeeze(0).detach().cpu())
            pred_position = self.get_bio_positions(bio_res=bio_result, input_id_int=input_id_int, input_prob=True, binary_mode=True, ignore_padding_token=False) #预测的span的下标索引

            # (1, lm_size )
            lm_clsss = lm_last_hidden_states[:, 0] #seq的第一个 最后的隐层状态中获取第一个 token（通常是 [CLS] token 的表示），形状为 (1, lm_size)，通常用于分类任务。
            # (1, lm_size + 1)
            lm_clsss = torch.cat([lm_clsss, torch.ones((1, 1), dtype=torch.float, device=device) * time_step], dim=1)
            # (1, node_size)
            lm_clss = self.lm_cls_hidden_linear(lm_clsss)
            # (1, seq_length, lm_size + 1) 创建一个新的张量，其形状与 lm_last_hidden_states 兼容，并填充一个标量值。将这个新张量与原始隐藏状态连接，扩展其特征维度。最终得到的结果 是一个包含了额外时间步信息的隐藏状态张量。
            lm_last_hidden_states = torch.cat([lm_last_hidden_states, torch.ones((1, lm_last_hidden_states.size(1), 1), dtype=torch.float, device=device) * time_step], dim=2)
            # (1, seq_length, node_size)
            lm_last_hidden_states = self.lm_hidden_linear(lm_last_hidden_states)
            # (seq_length, node_size)
            lm_hidden_state = lm_last_hidden_states.squeeze(0) #
            lm_clss_times.append(lm_clss)
            lm_hidden_state_times.append(lm_hidden_state)

            # --- bio post process, bio tag and bio position---
            if bios_ids is None: #表示进行评估
                ans_position = [] 
            else:
                # cpu bio ids [0, 0, 1, 5, 6]
                bio_ids = bio_ids.squeeze(0).detach().cpu().numpy().tolist()
                # cpu positions. [[start, end], [start, end]] 真实的position
                ans_position = self.get_bio_positions(bio_res=bio_ids, input_id_int=input_id_int, input_prob=False, binary_mode=False, ignore_padding_token=False)

            # --- train the model on the predicted span ---
            if bio_ids is not None: #训练train
                if use_mix_bio: #使用混合bio
                    position = pred_position + ans_position
                else:
                    position = ans_position
            else: #评估直接使用预测的作为position
                position = pred_position
            position_times.append(position)

        # --- event num predict ---
        # (cls_num, node_size)
        cls_for_event_num = torch.cat(lm_clss_times, dim=0)
        # (node_size)
        cls_for_event_num = torch.mean(cls_for_event_num, dim=0, keepdim=False)
        total_event_num_logit = self.cls_total_event_num_linear(cls_for_event_num) #预测事件总数概率分布
        total_event_num_pred = total_event_num_logit.detach().cpu().numpy().tolist()[0]
        total_event_num_pred = int(total_event_num_pred + 0.99999) #TODO：这里为什么加0.99999
        event_num_pred_result = total_event_num_pred
        total_event_num_index = self.num_proxy_slot
        num_all_proxy_slot = total_event_num_index
        # --- event num pred loss ---
        total_event_num = len(events_labels)
        total_event_num_label = torch.FloatTensor([total_event_num]).to(device)
        total_event_num_loss = self.mse_loss_fn(total_event_num_logit.view(-1), total_event_num_label.view(-1)) #事件个数损失函数

        # --- init the graph  构造图 ---
        # {(1, 2, 6, 9): [1, 5, 7], (5, 1, 6, 7): [2, 6, 10]}
        node_span_to_indexes = {} #span_node的index对应的input_ids 
        # {1: (1, 2, 6, 9), 2: (5, 1, 6, 7)}
        node_indexes_to_span = {}
        # [1, 3, 5, 7]
        cls_indexes = []
        # --- M-node  16个事件节点---
        proxy_all_indexes = list(range(num_all_proxy_slot))
        node_vector = self.initial_proxy_all_embedding[:total_event_num_index] #初始化代理事件节点的嵌入向量
        # ---- M-self ---
        # [[h, t], [h, t]]
        edge_index = [[i, i] for i in proxy_all_indexes] #自环 16
        # [0, 0, 0, 1, 1, 0]
        edge_type = [self.edge_type_table["M-self"]] * len(edge_index)
        # --- M-M relation M-all ---
        new_edge_index = [[head, tail] for tail in proxy_all_indexes for head in proxy_all_indexes if head != tail] #240=16*15
        new_edge_type = [self.edge_type_table['M-M']] * len(new_edge_index)
        edge_index += new_edge_index
        edge_type += new_edge_type

        # (proxy_slot_num, node_size) 16,512
        new_node_vector = [node_vector]
        current_index = node_vector.size(0) - 1
        for time_step in range(sentence_num): #构造其他节点节点
            # cpu [101, 102, 103]
            input_id_int = input_ids_int[time_step]
            lm_cls = lm_clss_times[time_step]
            position = position_times[time_step]
            lm_hidden_state = lm_hidden_state_times[time_step]
            # --- cls node  分类节点--- 用到了前面CLS的输出
            current_index += 1
            new_node_vector.append(lm_cls)
            cls_indexes.append(current_index)
            # --- C-M edge ---
            if 'C-M' in self.edge_type_table:
                new_edge_index = [[current_index, i] for i in proxy_all_indexes]
                new_edge_type = [self.edge_type_table['C-M']] * len(new_edge_index)
                edge_index += new_edge_index
                edge_type += new_edge_type
            # --- C-self --- 这个目前不在
            if 'C-self' in self.edge_type_table:
                new_edge_index = [[current_index, current_index]]
                new_edge_type = [self.edge_type_table['C-self']]
                edge_index += new_edge_index
                edge_type += new_edge_type
            # # --- C-C edge --- 这个目前不在
            if 'C-C' in self.edge_type_table:
                new_edge_index = [[current_index, i] for i in cls_indexes if current_index != i]
                new_edge_index += [[i, current_index] for i in cls_indexes if current_index != i]
                new_edge_type = [self.edge_type_table['C-C']] * len(new_edge_index)
                edge_index += new_edge_index
                edge_type += new_edge_type
            for pos in position:
                # cpu (101, 102, 103)
                if use_span_as_key:
                    span = tuple(input_id_int[pos[0]:pos[1]])
                else:
                    span = tuple([pos[0], pos[1]])
                # (span_length, node_size)
                span_hidden_state = lm_hidden_state[pos[0]:pos[1]]
                # (1, node_size)
                span_state = torch.mean(span_hidden_state, dim=0, keepdim=True) #span的平均state
                # --- span node  构造span节点，用到前面的position，然后去span的平均值作为span_node的嵌入向量表示---
                current_index += 1
                new_node_vector.append(span_state)
                # node_type.append(self.node_type_table['span'])
                node_indexes_to_span[current_index] = span
                if span not in node_span_to_indexes:
                    node_span_to_indexes[span] = [current_index] #span_node的index对应的input_ids
                else:
                    node_span_to_indexes[span].append(current_index)
                # --- S-M edge ---
                if 'S-M' in self.edge_type_table:
                    new_edge_index = [[current_index, i] for i in proxy_all_indexes] #mention---span
                    new_edge_type = [self.edge_type_table['S-M']] * len(new_edge_index)
                    edge_index += new_edge_index
                    edge_type += new_edge_type
                # --- S-C edge 这个没有 ---
                if 'S-C' in self.edge_type_table:
                    new_edge_index = [[current_index, i] for i in cls_indexes]
                    new_edge_type = [self.edge_type_table['S-C']] * len(new_edge_index)
                    edge_index += new_edge_index
                    edge_type += new_edge_type
                # --- C-S   这个没有  edge ---
                if 'C-S' in self.edge_type_table:
                    new_edge_index = [[i, current_index] for i in cls_indexes]
                    new_edge_type = [self.edge_type_table['C-S']] * len(new_edge_index)
                    edge_index += new_edge_index
                    edge_type += new_edge_type
                # --- S-self 这个没有 ---
                if 'S-self' in self.edge_type_table:
                    new_edge_index = [[current_index, current_index]]
                    new_edge_type = [self.edge_type_table['S-self']]
                    edge_index += new_edge_index
                    edge_type += new_edge_type
                # --- S-S edge  这个没有 ---
                if 'S-S' in self.edge_type_table:
                    if span in node_span_to_indexes and len(node_span_to_indexes[span]) > 1:
                        new_edge_index = [[current_index, i] for i in node_span_to_indexes[span] if current_index != i] + [[i, current_index] for i in node_span_to_indexes[span] if current_index != i]
                        new_edge_type = [self.edge_type_table['S-S']] * len(new_edge_index)
                        edge_index += new_edge_index
                        edge_type += new_edge_type

        # (proxy_slot_num+span_num, proxy_size)
        node_vector = torch.cat(new_node_vector, dim=0)
        assert len(edge_index) == len(edge_type) 
        assert node_vector.size(0) == current_index + 1

        # --- GCN ---
        # (proxy_slot_num+span_num, proxy_size)
        gcn_node_vector = self.gcn(x=node_vector,
                                   edge_index=torch.LongTensor(edge_index).to(device).t().contiguous(),
                                   edge_type=torch.LongTensor(edge_type).to(device),
                                   )

        # --- return records ---
        BIO_pred = torch.cat(bio_probs, dim=0).view(-1, self.num_BIO_tags).detach().cpu().numpy().tolist()
        records = {'BIO_pred': BIO_pred,
                   'loss_bio': loss_bio.item(),
                   } #解码的预测事件记录
        if len(node_span_to_indexes) == 0:
            loss = torch.FloatTensor([0]).to(device)
            loss += loss_bio
            if self.gradient_accumulation_steps is not None:
                loss = loss / self.gradient_accumulation_steps
            records.update({'loss': loss.item(),
                            'event_pred': [],
                            'error_report': "NodeSpanZero",
                            })
            return loss, records

        # (num_all_proxy_slot, node_size)
        proxy_slot = gcn_node_vector[:num_all_proxy_slot] #代理节点的gcn卷积后的向量

        # --- event type logit ---
        # (num_all_proxy_slot, num_event_type) 16,6  代理节点的事件类别
        event_type_logit = self.proxy_slot_event_type_linear(proxy_slot)

        # --- event relation logit --
        span_num = len(node_span_to_indexes) # span的数量
        max_individual_span_num = max([len(v) for k, v in node_span_to_indexes.items()]) #span对应的最多的个数,为了后续构造矩阵
        span_tensor_index_to_span = list(node_span_to_indexes.keys()) #span的input_ids
        span_tensor_span_to_index = {span_tensor_index_to_span[i]: i for i in range(span_num)} #input_ids2span_index
        # (span_num, individual_span_num, node_size) 构造一个不同span_num,这一组所有span对应的最大个数，node_size的矩阵
        span_tensor = torch.zeros((span_num, max_individual_span_num, self.node_size), dtype=torch.float, device=device)
        # (span_num, individual_span_num) mask矩阵
        span_tensor_mask = torch.ones((span_num, max_individual_span_num), dtype=torch.bool, device=device)
        for span, tensor_index in span_tensor_span_to_index.items():
            count = -1
            for span_state_index in node_span_to_indexes[span]:
                count += 1
                span_state = node_vector[span_state_index] #节点的span_state
                span_tensor[tensor_index, count] = span_state 
                span_tensor_mask[tensor_index, count] = False
        # (span_num, num_all_proxy_slot, node_size)
        proxy_slot_expand = proxy_slot.unsqueeze(0).expand(span_num, num_all_proxy_slot, self.node_size)
        # (span_num, num_all_proxy_slot, node_size)
        span_tensor, _ = self.proxy_span_attention(query=proxy_slot_expand, key=span_tensor, value=span_tensor, key_padding_mask=span_tensor_mask)
        # (num_all_proxy_slot, span_num, node_size*2)
        proxy_span_tensor = torch.cat([proxy_slot_expand, span_tensor], dim=2).transpose(0, 1)
        # (num_all_proxy_slot, span_num, num_event_relation)
        proxy_span_relation_logit = self.span_proxy_slot_relation_linear(proxy_span_tensor) #P_(a_i,k) 实体在与代理节点编码的事件相关的参数类型的概率分布

        # --- event probability result. This is the final for inference ---
        # (num_all_proxy_slot, num_event_type)
        event_type_prob = F.softmax(event_type_logit, dim=1)
        # (num_all_proxy_slot, span_num, num_event_relation)
        event_relation_prob = F.softmax(proxy_span_relation_logit, dim=2)

        # --- pack probability result ---
        # (num_all_proxy_slot, num_event_type)
        event_type_prob = event_type_prob.detach().cpu().numpy().tolist()
        # (num_all_proxy_slot, span_num, num_event_relation)
        event_relation_prob = event_relation_prob.detach().cpu().numpy().tolist()

        predict_events = [] #预测事件记录列表
        for j in range(num_all_proxy_slot):
            predict_event = {'EventType': event_type_prob[j]}
            for k in range(span_num):
                predict_event[span_tensor_index_to_span[k]] = event_relation_prob[j][k]
            predict_events.append(predict_event)

        # --- calculate the loss between null event and the memories events ---
        # (num_all_proxy_slot)
        null_event_type_label = torch.LongTensor([self.null_event_type_index]).to(device).expand(num_all_proxy_slot)
        # (span_num)
        null_event_relation_label = torch.LongTensor(torch.ones((span_num,), dtype=torch.long) * self.null_event_relation_index).to(device)
        # (num_all_proxy_slot, span_num)
        null_event_relation_label = null_event_relation_label.unsqueeze(0).expand(num_all_proxy_slot, span_num)

        # (num_all_proxy_slot)
        null_event_type_losses = self.ce_none_reduction_loss_fn(event_type_logit, null_event_type_label) #代理事件类型概率分布和空事件类型的交叉熵
        # (num_all_proxy_slot, span_num) 事件类型间关系的交叉熵损失
        null_event_relations_losses = self.ce_none_reduction_loss_fn(proxy_span_relation_logit.reshape(num_all_proxy_slot * span_num, self.num_event_relation), null_event_relation_label.reshape(num_all_proxy_slot * span_num)).view(num_all_proxy_slot, span_num)
        # (num_all_proxy_slot)
        null_event_relation_losses = torch.mean(null_event_relations_losses, dim=1, keepdim=False)

        events_label = events_labels

        records.update({'event_pred': predict_events,
                        })

        if len(events_label) == 0:
            total_null_type_loss = null_event_type_losses.mean()
            total_null_relation_loss = null_event_relation_losses.mean()
            loss = torch.FloatTensor([0]).to(device)
            loss += loss_bio
            loss += (total_null_type_loss + total_null_relation_loss)
            loss += total_event_num_loss

            records.update({'loss': loss.item(),
                            'error_report': 'NoLabelEvent',
                            })
            return loss, records

        # --- generate the label learnable tensor for each event to calculate the loss ---
        # (1) (span_num)
        events_label_type_to_index: Dict[str, list] = {x: [] for x in self.event_type_index_to_type_no_null}
        events_type_labels_tensors_list = []
        events_relation_labels_tensors_list = [] # 当前span_index对应的role_Index
        events_horizontal_role_labels_tensors_list = [] #当前role——index对应的span——index
        event_index = -1
        for event_label in events_label: #真实事件记录
            event_type_label_tensor = torch.LongTensor([event_label['EventType']]) #事件类型
            event_relation_label_tensor = torch.ones((span_num,), dtype=torch.long) * self.null_event_relation_index  #初始化长度为span_num，初始化元素为null对应的事件类型标签
            events_horizontal_role_label_tensor = torch.ones((self.num_event_relation,), dtype=torch.long) * -100  #初始化长度为role，初始值为-100
            for k, v in event_label.items():
                if k == 'EventType':
                    continue
                if k not in span_tensor_span_to_index:
                    # this should only happen when use not-gold bio tag
                    continue
                event_relation_label_tensor[span_tensor_span_to_index[k]] = v
                events_horizontal_role_label_tensor[v] = span_tensor_span_to_index[k]
            event_index += 1
            events_label_type_to_index[self.event_type_index_to_type[event_label['EventType']]].append(event_index)
            events_type_labels_tensors_list.append(event_type_label_tensor.to(device))
            events_relation_labels_tensors_list.append(event_relation_label_tensor.to(device))
            events_horizontal_role_labels_tensors_list.append(events_horizontal_role_label_tensor.to(device))
        event_num = len(events_type_labels_tensors_list)

        # --- calculate the loss between gold events and the memories events ---
        # (event_num) 真实事件类型列表
        events_type_labels_tensor = torch.cat(events_type_labels_tensors_list, dim=0)
        # (event_num, span_num)
        events_relation_labels_tensor = torch.cat([x.unsqueeze(0) for x in events_relation_labels_tensors_list], dim=0)
        # (event_num, num_event_relation) 代理节点和span之间的关系
        events_horizontal_role_labels_tensor = torch.cat([x.unsqueeze(0) for x in events_horizontal_role_labels_tensors_list], dim=0)

        # (event_num, num_all_proxy_slot, num_event_type) 预测事件类型概率分布
        event_type_logit_expand = event_type_logit.unsqueeze(0).expand(event_num, num_all_proxy_slot, self.num_event_type)
        # (event_num, num_all_proxy_slot) 真实事件类型扩展，为了后续计算交叉熵
        events_type_labels_expand = events_type_labels_tensor.unsqueeze(1).expand(event_num, num_all_proxy_slot)
        # (event_num, num_all_proxy_slot, span_num, num_event_relation) 代理节点和span之间的关系概率分布
        event_relation_logit_expand = proxy_span_relation_logit.unsqueeze(0).expand(event_num, num_all_proxy_slot, span_num, self.num_event_relation)
        # (event_num, num_all_proxy_slot, span_num) 每个事件（包括其代理节点和相关的跨度之间）之间的关系标签。 关注的是事件与事件之间的关系，主要用于理解事件之间的依赖或交互。
        events_relation_labels_expand = events_relation_labels_tensor.unsqueeze(1).expand(event_num, num_all_proxy_slot, span_num)
        # (event_num, num_all_proxy_slot, num_event_relation, span_num) 事件、span、角色预测关系概率分布
        event_relation_logit_expand_T = event_relation_logit_expand.transpose(2, 3)
        # (event_num, num_all_proxy_slot, num_event_relation) 对于某个代理槽，它包含了与该槽相关的所有事件类型的角色标签。 关注的是事件内的角色分配，即每个事件如何与其代理槽和角色相互作用。
        events_horizontal_role_labels_expand = events_horizontal_role_labels_tensor.unsqueeze(1).expand(event_num, num_all_proxy_slot, self.num_event_relation)

        # (event_num, num_all_proxy_slot) 事件类型的交叉熵损失
        event_type_losses = self.ce_none_reduction_loss_fn(event_type_logit_expand.reshape(event_num * num_all_proxy_slot, self.num_event_type), events_type_labels_expand.reshape(event_num * num_all_proxy_slot)).view(event_num, num_all_proxy_slot)
        # (event_num, num_all_proxy_slot, span_num) 事件间关系的损失
        event_relations_losses = self.ce_none_reduction_loss_fn(event_relation_logit_expand.reshape(event_num * num_all_proxy_slot * span_num, self.num_event_relation), events_relation_labels_expand.reshape(event_num * num_all_proxy_slot * span_num)).view(event_num, num_all_proxy_slot, span_num)
        # (event_num, num_all_proxy_slot)
        event_relation_losses = torch.mean(event_relations_losses, dim=2, keepdim=False)
        # (event_num, num_all_proxy_slot, num_event_relation) #在某个类型的slot下事件角色的交叉熵损失
        event_horizontal_role_losses = self.ce_none_reduction_loss_fn(event_relation_logit_expand_T.reshape(event_num * num_all_proxy_slot * self.num_event_relation, span_num), events_horizontal_role_labels_expand.reshape(event_num * num_all_proxy_slot * self.num_event_relation)).view(event_num, num_all_proxy_slot, self.num_event_relation)
        # (event_num, num_all_proxy_slot)
        event_horizontal_role_losses = torch.mean(event_horizontal_role_losses, dim=2, keepdim=False)

        # --- span-span relation --- 判断两实体是否属于同一个事件
        # for the_node_vector in [gcn_node_vector, node_vector]:
        ssr_pred = []  #span-span-relation_pred
        ssr_ans = []  #span-span-relation-answer 真实
        for the_node_vector in [node_vector]: #只执行一次，node_vector.size==role_num,node_size
            span_states_dict = {} #span2span_states
            for span, span_state_indexes in node_span_to_indexes.items():
                span_states = []
                for span_state_index in span_state_indexes:
                    span_state = the_node_vector[span_state_index]
                    span_states.append(span_state.unsqueeze(0))
                span_states = torch.cat(span_states, dim=0)
                span_states = torch.mean(span_states, dim=0, keepdim=False)
                span_states_dict[span] = span_states 
            ssr_index_to_span = list(span_states_dict.keys()) #len=span_num index2span  这两个都是input_ids和span_index的排序
            ssr_span_to_index = {ssr_index_to_span[i]: i for i in range(len(ssr_index_to_span))} #input_ids2index 
            span_all = [] #span_state 
            for span in ssr_index_to_span:
                span_all.append(span_states_dict[span].unsqueeze(0))
            # (span_num, span_state)
            span_all = torch.cat(span_all, dim=0)
            v_span_all = span_all.unsqueeze(1).expand(len(ssr_index_to_span), len(ssr_index_to_span), self.node_size) #span_num,span_num,node_size
            h_span_all = span_all.unsqueeze(0).expand(len(ssr_index_to_span), len(ssr_index_to_span), self.node_size)
            vh_span_all = torch.cat([v_span_all, h_span_all], dim=2) #span_num,span_num,2*node_size
            vh_span_all = self.span_span_relation_linear(vh_span_all) #判断两个实体是否属于同一个事件

            vh_span_all_label = torch.zeros((len(ssr_index_to_span), len(ssr_index_to_span)), dtype=torch.long)
            for event_label in events_label:
                for k1, v1 in event_label.items():
                    if k1 == 'EventType':
                        continue
                    if k1 not in ssr_span_to_index:
                        continue
                    k1_index = ssr_span_to_index[k1]
                    for k2, v2 in event_label.items():
                        if k2 == 'EventType':
                            continue
                        if k2 not in ssr_span_to_index:
                            continue
                        k2_index = ssr_span_to_index[k2]
                        vh_span_all_label[k1_index, k2_index] = 1 #同一事件种span间存在关系
            vh_span_all_label = vh_span_all_label.to(device)
            loss_span_span_relation = self.ce_normal_loss_fn(vh_span_all.view(-1, 2), vh_span_all_label.view(-1)) #span间关系交叉熵损失
            total_span_span_relation_loss = loss_span_span_relation

            ssr_pred += vh_span_all.view(-1, 2).detach().cpu().numpy().tolist()
            ssr_ans += vh_span_all_label.view(-1).detach().cpu().numpy().tolist()

        # --- loss ratio ---
        # (event_num, proxy_slot_num)
        event_loss_matrix = event_type_losses + event_relation_losses + event_horizontal_role_losses #整体损失
        # (proxy_slot_num)
        null_loss_matrix = (null_event_type_losses + null_event_relation_losses) #仅null事件损失
        # (1, proxy_slot_num)
        null_loss_matrix = null_loss_matrix.unsqueeze(0)
        if event_num < num_all_proxy_slot:#真实事件个数< proxy_slot_num 论文中提到的添加null事件伪节点作为negative标签补充个数
            padding_null_num = num_all_proxy_slot - event_num
            # (proxy_slot_num-event_num, proxy_slot_num)
            null_loss_matrix = null_loss_matrix.expand(padding_null_num, num_all_proxy_slot) #为null
            # (proxy_slot_num, proxy_slot_num)
            event_loss_matrix = torch.cat([event_loss_matrix, null_loss_matrix], dim=0) #所有事件损失
        # (max(event_num,proxy_slot_num), proxy_slot_num)
        losses_matrix_for_ordering = event_loss_matrix.detach().cpu().numpy()
        order_res_dict, min_order_loss = self.event_ordering(losses_matrix_for_ordering)
        # {event_id: proxy_slot_id}
        order_dict = {k: v for k, v in order_res_dict.items() if k < event_num} #保留最优匹配的前event_num个事件，也就是真实事件的个数

        # --- get total loss --- 将order_dict中与真实事件event_num个数匹配的代理事件
        event_positive_type_loss = 0
        event_positive_relation_loss = 0
        positive_total_num = 0
        null_total_num = num_all_proxy_slot
        for k, v in order_dict.items(): #k是max(event_num,proxy_slot_num)的序号，v是代表对应的代理槽ID。
            event_positive_type_loss += event_type_losses[k, v]
            event_positive_relation_loss += event_relation_losses[k, v] + event_horizontal_role_losses[k, v]
            positive_total_num += 1
            null_event_type_losses[v] = 0 # 清除和真实事件匹配的代理事件
            null_event_relation_losses[v] = 0
            null_total_num -= 1
        null_type_loss = torch.sum(null_event_type_losses)# 清除和真实事件匹配的代理事件之后，剩余的就是null事件类型损失
        null_relation_loss = torch.sum(null_event_relation_losses) #清除之后的关系

        one_type_loss = event_positive_type_loss if positive_total_num > 0 else 0
        one_relation_loss = event_positive_relation_loss if positive_total_num > 0 else 0
        one_null_type_loss = null_type_loss if null_total_num > 0 else 0
        one_null_relation_loss = null_relation_loss if null_total_num > 0 else 0

        total_type_loss = one_type_loss * self.neg_event_ratio_total
        total_relation_loss = one_relation_loss * self.neg_event_ratio_total
        total_null_type_loss = one_null_type_loss * self.pos_event_ratio_total # 因为正样本量少，null多，做一个平衡
        total_null_relation_loss = one_null_relation_loss * self.pos_event_ratio_total

        loss = torch.FloatTensor([0]).to(device)
        loss += loss_bio
        loss += (total_type_loss + total_relation_loss)
        loss += (total_null_type_loss + total_null_relation_loss)
        loss += total_event_num_loss
        loss += total_span_span_relation_loss

        if self.gradient_accumulation_steps is not None:
            loss = loss / self.gradient_accumulation_steps

        # --- return records ---
        records.update({'loss': loss.item(),
                        'error_report': '',
                        })
        return loss, records
