import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import numpy as np
import math

from pytorch_pretrained_bert import BertModel
from data_load import idx2trigger, argument2idx, all_triggers, all_arguments
from consts import NONE
from utils import find_triggers, get_trigger_loss_weights, get_arg_loss_weights


class Net(nn.Module):
    def __init__(self,
                 trigger_size=None,
                 entity_size=None,
                 all_postags=None,
                 postag_embedding_dim=50,
                 argument_size=None,
                 entity_embedding_dim=50,
                 device=torch.device("cpu")):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.entity_embed = MultiLabelEmbeddingLayer(num_embeddings=entity_size, embedding_dim=entity_embedding_dim, device=device)
        self.postag_embed = nn.Embedding(num_embeddings=all_postags, embedding_dim=postag_embedding_dim)
        #self.rnn = nn.LSTM(bidirectional=True, num_layers=1, input_size=768 + entity_embedding_dim, hidden_size=768 // 2, batch_first=True)

        # hidden_size = 768 + entity_embedding_dim + postag_embedding_dim
        mid_size = 4096
        hidden_size = 768

        self.trigger_output_mid_weights = torch.FloatTensor(mid_size, hidden_size)
        self.trigger_output_mid_bias = torch.FloatTensor(mid_size)
        self.trigger_output_weights = torch.FloatTensor(trigger_size, mid_size)
        self.trigger_output_bias = torch.FloatTensor(trigger_size)

        nn.init.normal_(self.trigger_output_mid_weights, std=0.02)
        nn.init.zeros_(self.trigger_output_mid_bias)
        nn.init.normal_(self.trigger_output_weights, std=0.02)
        nn.init.zeros_(self.trigger_output_bias)

        # 在原本的代码中未使用过fc1
        self.fc1 = nn.Sequential(
            # nn.Dropout(0.5),
            nn.Linear(hidden_size, hidden_size, bias=True),
            nn.ReLU(),
        )
        # unused
        self.fc_trigger = nn.Sequential(
            nn.Linear(hidden_size, trigger_size),
        )
        self.fc_argument = nn.Sequential(
            nn.Linear(hidden_size * 2, argument_size),
        )
        self.device = device

    def predict_triggers(self, tokens_x_2d, entities_x_3d, postags_x_2d, head_indexes_2d, triggers_y_2d, arguments_2d):
        tokens_x_2d = torch.LongTensor(tokens_x_2d).to(self.device)
        # postags_x_2d = torch.LongTensor(postags_x_2d).to(self.device)
        triggers_y_2d = torch.LongTensor(triggers_y_2d).to(self.device)
        head_indexes_2d = torch.LongTensor(head_indexes_2d).to(self.device)

        # postags_x_2d = self.postag_embed(postags_x_2d)
        # entity_x_2d = self.entity_embed(entities_x_3d)

        if self.training:
            self.bert.train()
            encoded_layers, _ = self.bert(tokens_x_2d)
            output_layer = encoded_layers[-1]
            dropout = nn.Dropout(0.1)
            output_layer = dropout(output_layer)
        else:
            self.bert.eval()
            with torch.no_grad():
                encoded_layers, _ = self.bert(tokens_x_2d)
                output_layer = encoded_layers[-1]
        # output_layer: [batch_size, seq_len, hidden_size]

        batch_size = output_layer.shape[0]
        seq_len = output_layer.shape[1]  # 每次train的时候seq_len都会变化
        hidden_size = output_layer.shape[2]

        for i in range(batch_size):
            output_layer[i] = torch.index_select(output_layer[i], 0, head_indexes_2d[i])

        X = output_layer
        X = torch.reshape(X, [-1, hidden_size])  # [batch_size * seq_len, hidden_size]
        X = torch.matmul(X, torch.transpose(self.trigger_output_mid_weights, 0, 1))  # [batch_size * seq_len, mid_size]
        X = torch.add(X, self.trigger_output_mid_bias)  # [batch_size * seq_len, mid_size]

        trigger_logits = torch.matmul(X, torch.transpose(self.trigger_output_weights, 0, 1))  # [batch_size * seq_len, argument_size]
        trigger_logits = torch.add(trigger_logits, self.trigger_output_bias)  # [batch_size * seq_len, trigger_size]
        trigger_logits = torch.reshape(trigger_logits, [-1, seq_len, trigger_logits.shape[-1]])  # [batch_size, seq_len, trigger_size]
        #trigger_logits = torch.clamp(trigger_logits, -1e-10, 1e+10)

        trigger_hat_2d = trigger_logits.argmax(-1)

        argument_hidden, argument_keys = [], []
        for i in range(batch_size):
            candidates = arguments_2d[i]['candidates']
            golden_entity_tensors = {}

            for j in range(len(candidates)):
                e_start, e_end, e_type_str = candidates[j]
                golden_entity_tensors[candidates[j]] = output_layer[i, e_start:e_end, ].mean(dim=0)

            predicted_triggers = find_triggers([idx2trigger[trigger] for trigger in trigger_hat_2d[i].tolist()])
            for predicted_trigger in predicted_triggers:
                t_start, t_end, t_type_str = predicted_trigger
                event_tensor = output_layer[i, t_start:t_end, ].mean(dim=0)
                for j in range(len(candidates)):
                    e_start, e_end, e_type_str = candidates[j]
                    entity_tensor = golden_entity_tensors[candidates[j]]

                    argument_hidden.append(torch.cat([event_tensor, entity_tensor]))
                    argument_keys.append((i, t_start, t_end, t_type_str, e_start, e_end, e_type_str))

        return trigger_logits, triggers_y_2d, trigger_hat_2d, argument_hidden, argument_keys

    def predict_arguments(self, argument_hidden, argument_keys, arguments_2d):
        argument_hidden = torch.stack(argument_hidden)
        argument_logits = self.fc_argument(argument_hidden)
        argument_hat_1d = argument_logits.argmax(-1)

        arguments_y_1d = []
        for i, t_start, t_end, t_type_str, e_start, e_end, e_type_str in argument_keys:
            a_label = argument2idx[NONE]
            if (t_start, t_end, t_type_str) in arguments_2d[i]['events']:
                for (a_start, a_end, a_type_idx) in arguments_2d[i]['events'][(t_start, t_end, t_type_str)]:
                    if e_start == a_start and e_end == a_end:
                        a_label = a_type_idx
                        break
            arguments_y_1d.append(a_label)

        arguments_y_1d = torch.LongTensor(arguments_y_1d).to(self.device)

        batch_size = len(arguments_2d)
        argument_hat_2d = [{'events': {}} for _ in range(batch_size)]
        for (i, st, ed, event_type_str, e_st, e_ed, entity_type), a_label in zip(argument_keys, argument_hat_1d.cpu().numpy()):
            if a_label == argument2idx[NONE]:
                continue
            if (st, ed, event_type_str) not in argument_hat_2d[i]['events']:
                argument_hat_2d[i]['events'][(st, ed, event_type_str)] = []
            argument_hat_2d[i]['events'][(st, ed, event_type_str)].append((e_st, e_ed, a_label))

        return argument_logits, arguments_y_1d, argument_hat_1d, argument_hat_2d


# Reused from https://github.com/lx865712528/EMNLP2018-JMEE
class MultiLabelEmbeddingLayer(nn.Module):
    def __init__(self,
                 num_embeddings=None, embedding_dim=None,
                 dropout=0.5, padding_idx=0,
                 max_norm=None, norm_type=2,
                 device=torch.device("cpu")):
        super(MultiLabelEmbeddingLayer, self).__init__()

        self.matrix = nn.Embedding(num_embeddings=num_embeddings,
                                   embedding_dim=embedding_dim,
                                   padding_idx=padding_idx,
                                   max_norm=max_norm,
                                   norm_type=norm_type)
        self.dropout = dropout
        self.device = device
        self.to(device)

    def forward(self, x):
        batch_size = len(x)
        seq_len = len(x[0])
        x = [self.matrix(torch.LongTensor(x[i][j]).to(self.device)).sum(0)
             for i in range(batch_size)
             for j in range(seq_len)]
        x = torch.stack(x).view(batch_size, seq_len, -1)

        if self.dropout is not None:
            return F.dropout(x, p=self.dropout, training=self.training)
        else:
            return x
