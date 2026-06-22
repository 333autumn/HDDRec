"""
原始kgcl模型代码
"""
import torch
import torch.nn.functional as F
from torch import nn

import world
from GAT import GAT
from dataloader import BasicDataset
from utils import _L2_loss_mean


class BasicModel(nn.Module):
    def __init__(self):
        super(BasicModel, self).__init__()

    def getUsersRating(self, users):
        raise NotImplementedError


class KGCL(BasicModel):
    def __init__(self,
                 config: dict,
                 dataset: BasicDataset,
                 kg_dataset):
        super(KGCL, self).__init__()
        self.config = config
        self.dataset: BasicDataset = dataset
        self.kg_dataset = kg_dataset
        self.__init_weight()
        self.gat = GAT(self.latent_dim, self.latent_dim,
                       dropout=0.4, alpha=0.2).train()

    def __init_weight(self):
        self.num_users = self.dataset.n_users
        self.num_items = self.dataset.m_items
        self.num_entities = self.kg_dataset.entity_count
        self.num_relations = self.kg_dataset.relation_count
        print("user:{}, item:{}, entity:{}".format(
            self.num_users, self.num_items, self.num_entities))
        self.latent_dim = self.config['latent_dim_rec']
        self.n_layers = self.config['lightGCN_n_layers']
        self.keep_prob = self.config['keep_prob']
        self.A_split = self.config['A_split']

        """
        加载compute_globalhidden相关参数
        """
        self.hidden_size = self.latent_dim
        self.linear_one = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_two = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_three = nn.Linear(self.hidden_size, 1, bias=False)
        self.linear_four = nn.Linear(self.hidden_size, 1, bias=False)
        self.linear_transform = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)

        # 均匀分布初始化
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim, padding_idx=-1)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim, padding_idx=-1)
        self.embedding_entity = torch.nn.Embedding(
            num_embeddings=self.num_entities + 1, embedding_dim=self.latent_dim, padding_idx=-1)
        self.embedding_relation = torch.nn.Embedding(
            num_embeddings=self.num_relations + 1, embedding_dim=self.latent_dim, padding_idx=-1)
        initrange = 0.5 / self.latent_dim
        nn.init.uniform_(self.embedding_user.weight, -initrange, initrange)
        nn.init.uniform_(self.embedding_item.weight, -initrange, initrange)
        nn.init.uniform_(self.embedding_entity.weight, -initrange, initrange)
        nn.init.uniform_(self.embedding_relation.weight, -initrange, initrange)

        # relation weights
        self.W_R = nn.Parameter(torch.Tensor(
            self.num_relations, self.latent_dim, self.latent_dim))
        nn.init.xavier_uniform_(self.W_R, gain=nn.init.calculate_gain('relu'))

        if self.config['pretrain'] == 0:
            world.cprint('use NORMAL distribution UI')
            nn.init.normal_(self.embedding_user.weight, std=0.1)
            nn.init.normal_(self.embedding_item.weight, std=0.1)
            world.cprint('use NORMAL distribution ENTITY')
            nn.init.normal_(self.embedding_entity.weight, std=0.1)
            nn.init.normal_(self.embedding_relation.weight, std=0.1)
        else:
            self.embedding_user.weight.data.copy_(
                torch.from_numpy(self.config['user_emb']))
            self.embedding_item.weight.data.copy_(
                torch.from_numpy(self.config['item_emb']))
            print('use pretarined data')
        self.f = nn.Sigmoid()
        self.Graph = self.dataset.getSparseGraph()
        # self.ItemNet = self.kg_dataset.get_item_net_from_kg(self.num_items)
        self.kg_dict, self.item2relations = self.kg_dataset.get_kg_dict(
            self.num_items)
        print(f"KGCL is ready to go!")

    # todo 自适应丢弃策略
    def __dropout_x(self, x, keep_prob):
        """
        对输入的稀疏图 x 执行基于节点度的自适应边丢弃操作。

        该方法通过在内部动态计算节点度，避免了对外部 self.degrees 的依赖，
        从而解决了维度不匹配的问题，并实现了一个健壮的丢弃策略。

        参数:
            x (torch.sparse.FloatTensor): 输入的稀疏图，形状为 (num_nodes, num_nodes)。
            keep_prob (float): 全局的边保持概率。

        返回:
            torch.sparse.FloatTensor: 经过丢弃操作后的稀疏图。
        """
        size = x.size()
        index = x.indices().t()  # 转置得到 (num_edges, 2)，每一行是一条边的 (源节点, 目标节点) 索引
        values = x.values()  # (num_edges,)，每条边的权重

        # --- 步骤 1: 动态计算节点度 ---
        # 使用输入图 x 的索引信息来实时计算每个节点的度，确保数据一致性
        num_nodes = size[0]  # 图中节点的总数
        row = x.indices()[0]  # 所有边的源节点索引
        col = x.indices()[1]  # 所有边的目标节点索引

        # 计算每个节点的出度 (作为源节点的次数)
        degrees_out = torch.bincount(row, minlength=num_nodes)
        # 计算每个节点的入度 (作为目标节点的次数)
        degrees_in = torch.bincount(col, minlength=num_nodes)
        # 计算每个节点的总度 (出度 + 入度)
        degrees = degrees_out + degrees_in

        # --- 步骤 2: 计算自适应丢弃概率 ---
        # 根据每条边的源节点度，计算该边的丢弃概率
        max_degree = torch.max(degrees).float()  # 图中的最大度，用于归一化
        # 获取每条边的源节点对应的度
        edge_source_degrees = degrees[row]
        # 计算每条边的重要性分数 (0到1之间)
        probabilities = edge_source_degrees / max_degree
        # 裁剪概率，避免过小，增加模型鲁棒性
        probabilities = torch.clamp(probabilities, min=0.1)

        # 根据重要性分数和全局 keep_prob 计算调整后的保持概率
        # 注意: 这里的公式可能需要根据具体论文或需求调整
        # 当前公式: 重要性越高的边，被丢弃的概率越大
        adjusted_keep_prob = 1 - (1 - keep_prob) * probabilities

        # --- 步骤 3: 生成随机掩码并应用 ---
        # 生成与边数量相同的随机数
        random_tensor = torch.rand(len(values), device=values.device)
        # 根据调整后的保持概率生成丢弃掩码
        # 如果 random_tensor 中的值小于 adjusted_keep_prob，则该边被保留 (mask为True)
        dropout_mask = random_tensor < adjusted_keep_prob

        # 应用掩码，筛选出被保留的边的索引和值
        retained_indices = index[dropout_mask].t()  # 转置回 (2, num_retained_edges)
        retained_values = values[dropout_mask]

        # 对保留下来的边的权重进行缩放，以补偿被丢弃的边
        # 缩放因子是 1 / (该边的实际保持概率)
        # 注意: 这里需要处理 adjusted_keep_prob 为 0 的情况，避免除以零
        # 由于有 clamp(min=0.1)，adjusted_keep_prob 不会为0，但保留此处的健壮性思考
        scaling_factors = 1.0 / adjusted_keep_prob[dropout_mask]
        retained_values = retained_values * scaling_factors

        # --- 步骤 4: 重建稀疏张量 ---
        # 使用被保留的边的索引和值，重建一个新的稀疏图
        g = torch.sparse.FloatTensor(retained_indices, retained_values, size)
        return g

    def __dropout(self, keep_prob):
        if self.A_split:
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.Graph, keep_prob)
        return graph

    def view_computer_all(self, g_droped, kg_droped):
        """
        propagate methods for contrastive lightGCN
        """
        users_emb = self.embedding_user.weight
        items_emb = self.cal_item_embedding_from_kg(kg_droped)
        all_emb = torch.cat([users_emb, items_emb])
        #   torch.split(all_emb , [self.num_users, self.num_items])
        embs = [all_emb]
        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items

    def view_computer_ui(self, g_droped):
        """
        propagate methods for contrastive lightGCN
        """
        users_emb = self.embedding_user.weight
        items_emb = self.cal_item_embedding_from_kg(self.kg_dict)
        all_emb = torch.cat([users_emb, items_emb])
        #   torch.split(all_emb , [self.num_users, self.num_items])
        embs = [all_emb]
        for layer in range(self.n_layers):
            if self.A_split:
                temp_emb = []
                for f in range(len(g_droped)):
                    temp_emb.append(torch.sparse.mm(g_droped[f], all_emb))
                side_emb = torch.cat(temp_emb, dim=0)
                all_emb = side_emb
            else:
                all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items

    def computer(self):
        """
        propagate methods for lightGCN
        """
        users_emb = self.embedding_user.weight
        items_emb = self.cal_item_embedding_from_kg(self.kg_dict)
        all_emb = torch.cat([users_emb, items_emb])
        embs = [all_emb]
        # 启动时设置参数--dropout=1
        if self.config['dropout']:
            if self.training:
                g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.Graph
        else:
            g_droped = self.Graph

        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        # print(embs.size())
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items

    def getUsersRating(self, users):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating

    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego

    def bpr_loss(self, users, pos, neg):
        (users_emb, pos_emb, neg_emb,
         userEmb0, posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())
        reg_loss = (1 / 2) * (userEmb0.norm(2).pow(2) +
                              posEmb0.norm(2).pow(2) +
                              negEmb0.norm(2).pow(2)) / float(len(users))
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)

        # mean or sum
        loss = torch.sum(
            torch.nn.functional.softplus(-(pos_scores - neg_scores)))
        if (torch.isnan(loss).any().tolist()):
            print("user emb")
            print(userEmb0)
            print("pos_emb")
            print(posEmb0)
            print("neg_emb")
            print(negEmb0)
            print("neg_scores")
            print(neg_scores)
            print("pos_scores")
            print(pos_scores)
            return None
        return loss, reg_loss

    # todo 计算全局隐藏状态
    def compute_globalhidden(self, hidden):  # tensor:(19386,32)
        # 创建掩码，根据实际数据生成
        device = hidden.device
        mask = torch.ones(hidden.shape[0], hidden.shape[1], dtype=torch.float32).to(device)

        if hidden.dim() == 2:  # 如果输入只有两个维度
            hidden = hidden.unsqueeze(1)  # 添加一个伪序列维度，(batch_size, 1, latent_size)
            mask = mask.unsqueeze(1)  # 掩码维度也需要扩展

        # 计算隐藏状态的线性变换
        q2 = self.linear_two(hidden)  # (batch_size, seq_length, latent_size)
        # new part
        # adding attention
        # 使用线性变换和 sigmoid 函数计算注意力权重 alpha
        alpha = self.linear_three(torch.sigmoid(q2))  # tensor:(10001,1,1)
        # 将 alpha 和隐藏状态 hidden 按照掩码 mask 进行加权求和，得到初始的全局隐藏表示 a。
        a = torch.sum(alpha * hidden * mask.view(mask.shape[0], -1, 1).float(), 1)
        # 将初始全局表示 a 经过线性变换得到 q1。
        q1 = self.linear_one(a).view(a.shape[0], 1, a.shape[1])
        # q1与之前计算的 q2 相加后，通过 sigmoid 和线性变换得到新的注意力权重 beta。
        beta = self.linear_four(torch.sigmoid(q1 + q2))
        # 使用 beta 对隐藏状态 hidden 进行加权求和，生成新的全局隐藏表示 newa
        newa = torch.sum(beta * hidden * mask.view(mask.shape[0], -1, 1).float(), 1)
        a = self.linear_transform(torch.cat([a, newa], 1))  # (10001,160)
        return a

    def calc_kg_loss_transE(self, h, r, pos_t, neg_t):
        """
        h:      (kg_batch_size)
        r:      (kg_batch_size)
        pos_t:  (kg_batch_size)
        neg_t:  (kg_batch_size)
        """
        r_embed = self.embedding_relation(
            r)  # (kg_batch_size, relation_dim)
        # (kg_batch_size, entity_dim)
        h_embed = self.embedding_item(h)
        pos_t_embed = self.embedding_entity(
            pos_t)  # (kg_batch_size, entity_dim)
        neg_t_embed = self.embedding_entity(
            neg_t)  # (kg_batch_size, entity_dim)

        # Equation (1)
        pos_score = torch.sum(
            torch.pow(h_embed + r_embed - pos_t_embed, 2), dim=1)  # (kg_batch_size)
        neg_score = torch.sum(
            torch.pow(h_embed + r_embed - neg_t_embed, 2), dim=1)  # (kg_batch_size)

        # Equation (2)
        kg_loss = (-1.0) * F.logsigmoid(neg_score - pos_score)
        kg_loss = torch.mean(kg_loss)

        l2_loss = _L2_loss_mean(h_embed) + _L2_loss_mean(r_embed) + \
                  _L2_loss_mean(pos_t_embed) + _L2_loss_mean(neg_t_embed)

        loss = kg_loss + 1e-3 * l2_loss
        # loss = kg_loss
        return loss

    def calc_kg_loss(self, h, r, pos_t, neg_t):
        """
        h:      (kg_batch_size)
        r:      (kg_batch_size)
        pos_t:  (kg_batch_size)
        neg_t:  (kg_batch_size)
        """
        r_embed = self.embedding_relation(
            r)  # (kg_batch_size, relation_dim)
        # (kg_batch_size, entity_dim, relation_dim)
        W_r = self.W_R[r]

        # (kg_batch_size, entity_dim)
        h_embed = self.embedding_item(h)
        pos_t_embed = self.embedding_entity(
            pos_t)  # (kg_batch_size, entity_dim)
        neg_t_embed = self.embedding_entity(
            neg_t)  # (kg_batch_size, entity_dim)

        r_mul_h = torch.bmm(h_embed.unsqueeze(1), W_r).squeeze(
            1)  # (kg_batch_size, relation_dim)
        r_mul_pos_t = torch.bmm(pos_t_embed.unsqueeze(1), W_r).squeeze(
            1)  # (kg_batch_size, relation_dim)
        r_mul_neg_t = torch.bmm(neg_t_embed.unsqueeze(1), W_r).squeeze(
            1)  # (kg_batch_size, relation_dim)

        # Equation (1)
        pos_score = torch.sum(
            torch.pow(r_mul_h + r_embed - r_mul_pos_t, 2), dim=1)  # (kg_batch_size)
        neg_score = torch.sum(
            torch.pow(r_mul_h + r_embed - r_mul_neg_t, 2), dim=1)  # (kg_batch_size)

        # Equation (2)
        kg_loss = (-1.0) * F.logsigmoid(neg_score - pos_score)
        kg_loss = torch.mean(kg_loss)

        l2_loss = _L2_loss_mean(r_mul_h) + _L2_loss_mean(r_embed) + \
                  _L2_loss_mean(r_mul_pos_t) + _L2_loss_mean(r_mul_neg_t)
        loss = kg_loss + 1e-3 * l2_loss
        # loss = kg_loss
        return loss

    def cal_item_embedding_gat(self, kg: dict):
        item_embs = self.embedding_item(torch.IntTensor(
            list(kg.keys())).to(world.device))  # item_num, emb_dim
        # item_num, entity_num_each
        item_entities = torch.stack(list(kg.values()))
        # item_num, entity_num_each, emb_dim
        entity_embs = self.embedding_entity(item_entities)
        # item_num, entity_num_each
        padding_mask = torch.where(item_entities != self.num_entities, torch.ones_like(
            item_entities), torch.zeros_like(item_entities)).float()
        return self.gat(item_embs, entity_embs, padding_mask)

    def cal_item_embedding_rgat(self, kg: dict):
        item_embs = self.embedding_item(torch.IntTensor(
            list(kg.keys())).to(world.device))  # item_num, emb_dim
        # item_num, entity_num_each
        item_entities = torch.stack(list(kg.values()))
        item_relations = torch.stack(list(self.item2relations.values()))
        # item_num, entity_num_each, emb_dim
        entity_embs = self.embedding_entity(item_entities)
        relation_embs = self.embedding_relation(
            item_relations)  # item_num, entity_num_each, emb_dim
        # w_r = self.W_R[relation_embs] # item_num, entity_num_each, emb_dim, emb_dim
        # item_num, entity_num_each
        padding_mask = torch.where(item_entities != self.num_entities, torch.ones_like(
            item_entities), torch.zeros_like(item_entities)).float()
        return self.gat.forward_relation(item_embs, entity_embs, relation_embs, padding_mask)

    def cal_item_embedding_from_kg(self, kg: dict):
        if kg is None:
            kg = self.kg_dict

        if (world.kgcn == "GAT"):
            return self.cal_item_embedding_gat(kg)
        elif world.kgcn == "RGAT":
            return self.cal_item_embedding_rgat(kg)
        elif (world.kgcn == "MEAN"):
            return self.cal_item_embedding_mean(kg)
        elif (world.kgcn == "NO"):
            return self.embedding_item.weight

    def cal_item_embedding_mean(self, kg: dict):
        item_embs = self.embedding_item(torch.IntTensor(
            list(kg.keys())).to(world.device))  # item_num, emb_dim
        # item_num, entity_num_each
        item_entities = torch.stack(list(kg.values()))
        # item_num, entity_num_each, emb_dim
        entity_embs = self.embedding_entity(item_entities)
        # item_num, entity_num_each
        padding_mask = torch.where(item_entities != self.num_entities, torch.ones_like(
            item_entities), torch.zeros_like(item_entities)).float()
        # padding为0
        entity_embs = entity_embs * \
                      padding_mask.unsqueeze(-1).expand(entity_embs.size())
        # item_num, emb_dim
        entity_embs_sum = entity_embs.sum(1)
        entity_embs_mean = entity_embs_sum / \
                           padding_mask.sum(-1).unsqueeze(-1).expand(entity_embs_sum.size())
        # replace nan with zeros
        entity_embs_mean = torch.nan_to_num(entity_embs_mean)
        # item_num, emb_dim
        return item_embs + entity_embs_mean

    def forward(self, users, items):
        # compute embedding
        all_users, all_items = self.computer()
        # print('forward')
        # all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]

        # todo 对两种嵌入进行处理
        users_emb = self.compute_globalhidden(users_emb)
        items_emb = self.compute_globalhidden(items_emb)

        inner_pro = torch.mul(users_emb, items_emb)
        gamma = torch.sum(inner_pro, dim=1)
        return gamma
