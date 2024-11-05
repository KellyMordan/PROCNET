import torch
import torch.nn.functional as F
from torch import nn
from geomloss import SamplesLoss
from DocEE_proxy_node_model import DocEEGNNModelHN

class ProxyNodeOTModel(nn.Module):
    def __init__(self, config, preparer):
        super(ProxyNodeOTModel, self).__init__()
        self.config = config
        self.preparer = preparer
        self.node_size = config.node_size
        self.num_proxy_slot = config.proxy_slot_num
        self.num_event_type = len(preparer.event_type_type_to_index)
        self.num_event_relation = len(preparer.event_role_relation_to_index)
        self.null_event_type_index = preparer.event_type_type_to_index['Null']
        self.null_event_relation_index = preparer.event_role_relation_to_index['Null']

        self.gcn = DocEEGNNModelHN(node_size=self.node_size, num_relations=4, dropout_ratio=0.15)
        self.proxy_slot_event_type_linear = nn.Linear(self.node_size, self.num_event_type)
        self.proxy_span_relation_linear = nn.Linear(self.node_size, self.num_event_relation)
        
        # OT相关参数
        self.m = torch.randn(self.num_proxy_slot, requires_grad=True, device="cuda")
        self.WDLoss = SamplesLoss(loss="sinkhorn", p=1, blur=0.01, backend="tensorized")

        # 损失函数
        self.ce_none_reduction_loss_fn = nn.CrossEntropyLoss(reduction='none')
        self.ce_normal_loss_fn = nn.CrossEntropyLoss()
        
    def forward(self, node_vector, gold_events, gold_roles, edge_index, edge_type):
        device = next(self.parameters()).device

        # --- GCN ---
        gcn_node_vector = self.gcn(
            x=node_vector,
            edge_index=edge_index,
            edge_type=edge_type,
        )

        # (num_proxy_slot, node_size)
        proxy_slot = gcn_node_vector[:self.num_proxy_slot]

        # Event type prediction (linear layer)
        event_type_logit = self.proxy_slot_event_type_linear(proxy_slot)
        event_type_prob = F.softmax(event_type_logit, dim=1)
        
        # Role prediction (linear layer)
        role_logits = self.proxy_span_relation_linear(proxy_slot)  # (num_proxy_slot, num_event_relation)
        role_probs = F.softmax(role_logits, dim=1)
        
        # 优化向量 m (通过Gumbel-Softmax)
        pr = torch.sigmoid(self.m.view(-1, 1))
        b = F.gumbel_softmax(torch.cat([pr, 1 - pr], dim=1), tau=0.5, hard=True)[:, 0]

        # 构造保留的代理节点
        selected_proxy_nodes = proxy_slot[b == 1]
        gold_event_tensor = torch.stack(gold_events).to(device)

        # 计算Hausdorff距离
        wd_loss = self.WDLoss(
            selected_proxy_nodes.unsqueeze(0),  # 预测的事件集合
            gold_event_tensor.unsqueeze(0)      # 真实的事件集合
        )

        # Null事件和非Null事件的损失分开计算
        null_event_type_label = torch.LongTensor([self.null_event_type_index]).to(device).expand(self.num_proxy_slot)
        null_event_relation_label = torch.LongTensor([self.null_event_relation_index]).to(device).expand(self.num_proxy_slot)

        # 计算Null事件的损失
        null_event_type_losses = self.ce_none_reduction_loss_fn(event_type_logit, null_event_type_label)
        null_event_relation_losses = self.ce_none_reduction_loss_fn(role_logits, null_event_relation_label)
        null_event_loss = torch.sum(null_event_type_losses) + torch.sum(null_event_relation_losses)

        # 计算非Null事件的损失
        gold_event_type_tensor = torch.stack([event['EventType'] for event in gold_events]).to(device)
        gold_role_tensor = torch.stack(gold_roles).to(device)

        event_type_loss = self.ce_normal_loss_fn(event_type_logit, gold_event_type_tensor)
        role_loss = self.ce_normal_loss_fn(role_logits, gold_role_tensor)

        # 总损失
        total_loss = wd_loss + event_type_loss + role_loss + null_event_loss
        return total_loss, event_type_prob

# 使用实例化和前向传播
config = DocEEConfig()  # 假设你有合适的配置类
preparer = DocEEPreparer()  # 假设你有合适的数据准备类
model = ProxyNodeOTModel(config, preparer)

# 示例输入
node_vector = torch.randn((config.proxy_slot_num + 10, config.node_size)).cuda()
edge_index = torch.randint(0, config.proxy_slot_num, (2, 20)).long().cuda()
edge_type = torch.randint(0, 4, (20,)).long().cuda()
gold_events = [{'EventType': torch.randint(0, config.num_event_type, (1,)).item()} for _ in range(5)]  # 假设有5个真实事件
gold_roles = [torch.randint(0, config.num_event_relation, (config.num_proxy_slot,)) for _ in range(5)]  # 假设每个事件有角色信息

# 前向传播调用
total_loss, event_type_prob = model(node_vector, gold_events, gold_roles, edge_index, edge_type)
print("Total Loss:", total_loss.item())
