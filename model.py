import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_max_pool as gmp
from torch.nn.utils.rnn import pad_sequence
from torch.nn import MultiheadAttention
import math
from fairseq.models import FairseqIncrementalDecoder
from fairseq.modules import TransformerDecoderLayer, TransformerEncoderLayer
from einops.layers.torch import Rearrange
from typing import Optional, Dict
from utils import Tokenizer

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class Namespace:
    def __init__(self, argvs):
        for k, v in argvs.items():
            setattr(self, k, v)

class TransformerEncoder(nn.Module):
    def __init__(self, dim, ff_dim, num_head, num_layer):
        super().__init__()
        self.layer = nn.ModuleList([
            TransformerEncoderLayer(Namespace({
                'encoder_embed_dim': dim,
                'encoder_attention_heads': num_head,
                'attention_dropout': 0.1,
                'dropout': 0.1,
                'encoder_normalize_before': True,
                'encoder_ffn_embed_dim': ff_dim,
            })) for _ in range(num_layer)
        ])
        self.layer_norm = nn.LayerNorm(dim)

    def forward(self, x, encoder_padding_mask=None):
        for layer in self.layer:
            x = layer(x, encoder_padding_mask)
        x = self.layer_norm(x)
        return x

class Encoder(torch.nn.Module):
    def __init__(self, Drug_Features, dropout, Final_dim):
        super(Encoder, self).__init__()
        self.hidden_dim = 376
        self.GraphConv1 = GCNConv(Drug_Features, Drug_Features * 2)
        self.GraphConv2 = GCNConv(Drug_Features * 2, Drug_Features * 3)
        self.GraphConv3 = GCNConv(Drug_Features * 3, Drug_Features * 4)
        self.GEAFF1 = nn.Sequential(
            nn.Linear(Drug_Features * 4 + Drug_Features * 4, Drug_Features * 4),
            nn.Sigmoid()
        )
        self.GEAFF2 = nn.Sequential(
            nn.Linear(Drug_Features * 4 + Drug_Features * 4, Drug_Features * 4),
            nn.Sigmoid()
        )
        self.s1 = nn.Linear(Drug_Features * 2, Drug_Features * 4)
        self.s2 = nn.Linear(Drug_Features * 3, Drug_Features * 4)
        self.cond = nn.Linear(96 * 107, self.hidden_dim)
        self.cond2 = nn.Linear(451, self.hidden_dim)
        self.attention_gcn = nn.Sequential(
            nn.Linear(Drug_Features * 4, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, 1)
        )
        self.attention_pmvo = nn.Sequential(
            nn.Linear(Drug_Features * 4, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, 1)
        )
        self.mean = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        self.var = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        self.Drug_FCs = nn.Sequential(
            nn.Linear(Drug_Features * 4, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, Final_dim)
        )
        self.Relu_activation = nn.ReLU()
        self.pp_seg_encoding = nn.Parameter(torch.randn(376))
        self.dropout = nn.Dropout(dropout)

    def reparameterize(self, z_mean, logvar, batch, con, a):
        z_log_var = -torch.abs(logvar)
        kl_loss = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp()) / 64
        epsilon = torch.randn_like(z_mean).to(z_mean.device)
        z_ = z_mean + torch.exp(z_log_var / 2) * epsilon
        con = con.view(-1, 96 * 107)
        con_embedding = self.cond(con)
        z_ = z_ + con_embedding + a
        return z_, kl_loss

    def process_p(self, node_features, num_nodes, batch_size):
        d_node_features = pad_sequence(torch.split(node_features, num_nodes.tolist()), 
                                      batch_first=False, padding_value=-999)
        padded_sequence = d_node_features.new_ones((d_node_features.shape[0], 
                                                   d_node_features.shape[1], 
                                                   d_node_features.shape[2])) * -999
        padded_sequence[:d_node_features.shape[0], :, :] = d_node_features
        d_node_features = padded_sequence
        padding_mask = (d_node_features[:, :, 0].T == -999).bool()
        padded_sequence_with_encoding = d_node_features + self.pp_seg_encoding
        return padded_sequence_with_encoding, padding_mask

    def forward(self, data, con):
        x, edge_index, batch, num_nodes, affinity = data.x, data.edge_index, data.batch, data.c_size, data.y
        a = affinity.view(-1, 1)
        GCNConv1 = self.GraphConv1(x, edge_index)
        GCNConv1 = self.Relu_activation(GCNConv1)
        GCNConv1 = self.dropout(GCNConv1)
        GCNConv2 = self.GraphConv2(GCNConv1, edge_index)
        GCNConv2 = self.Relu_activation(GCNConv2)
        GCNConv2 = self.dropout(GCNConv2)
        GCNConv3 = self.GraphConv3(GCNConv2, edge_index)
        GCNConv3 = self.Relu_activation(GCNConv3)
        GCNConv3 = self.dropout(GCNConv3)
        s1 = self.s1(GCNConv1)
        s2 = self.s2(GCNConv2)
        GEAFF1_weight = self.GEAFF1(torch.cat([GCNConv3, s1], dim=1))
        GEAFF2_weight = self.GEAFF2(torch.cat([GCNConv3, s2], dim=1))
        GCN_out = GCNConv3 + GEAFF1_weight * s1 + GEAFF2_weight * s2
        x = self.Relu_activation(GCN_out)
        fused_features = x
        d_sequence, Mask = self.process_p(fused_features, num_nodes, batch)
        mu = self.mean(d_sequence)
        logvar = self.var(d_sequence)
        AOUT, kl_loss = self.reparameterize(mu, logvar, batch, con, a)
        x2 = gmp(fused_features, batch)
        Drug_feature = self.Drug_FCs(x2)
        return d_sequence, AOUT, Mask, Drug_feature, kl_loss

class Decoder(nn.Module):
    def __init__(self, dim, ff_dim, num_head, num_layer):
        super().__init__()
        self.layer = nn.ModuleList([
            TransformerDecoderLayer(Namespace({
                'decoder_embed_dim': dim,
                'decoder_attention_heads': num_head,
                'attention_dropout': 0.1,
                'dropout': 0.1,
                'decoder_normalize_before': True,
                'decoder_ffn_embed_dim': ff_dim,
            })) for _ in range(num_layer)
        ])
        self.layer_norm = nn.LayerNorm(dim)

    def forward(self, x, mem, x_mask=None, x_padding_mask=None, mem_padding_mask=None):
        for layer in self.layer:
            x = layer(x, mem,
                      self_attn_mask=x_mask, self_attn_padding_mask=x_padding_mask,
                      encoder_padding_mask=mem_padding_mask)[0]
        x = self.layer_norm(x)
        return x

    @torch.jit.export
    def forward_one(self,
                    x: torch.Tensor,
                    mem: torch.Tensor,
                    incremental_state: Optional[Dict[str, Dict[str, Optional[torch.Tensor]]]],
                    mem_padding_mask: torch.BoolTensor = None,
                    ) -> torch.Tensor:
        x = x[-1:]
        for layer in self.layer:
            x = layer(x, mem, incremental_state=incremental_state, encoder_padding_mask=mem_padding_mask)[0]
        x = self.layer_norm(x)
        return x

class GatedCNN(nn.Module):
    def __init__(self, Protein_Features, Num_Filters, Embed_dim, Final_dim, K_size):
        super(GatedCNN, self).__init__()
        self.Protein_Embed = nn.Embedding(Protein_Features + 1, Embed_dim)
        self.Protein_Conv1 = nn.Conv1d(in_channels=1000, out_channels=Num_Filters, kernel_size=K_size)
        self.Protein_Gate1 = nn.Conv1d(in_channels=1000, out_channels=Num_Filters, kernel_size=K_size)
        self.Protein_Conv2 = nn.Conv1d(in_channels=Num_Filters, out_channels=Num_Filters * 2, kernel_size=K_size)
        self.Protein_Gate2 = nn.Conv1d(in_channels=Num_Filters, out_channels=Num_Filters * 2, kernel_size=K_size)
        self.Protein_Conv3 = nn.Conv1d(in_channels=Num_Filters * 2, out_channels=Num_Filters * 3, kernel_size=K_size)
        self.Protein_Gate3 = nn.Conv1d(in_channels=Num_Filters * 2, out_channels=Num_Filters * 3, kernel_size=K_size)
        self.relu = nn.ReLU()
        self.Protein_FC = nn.Linear(96 * 107, Final_dim)

    def forward(self, data):
        target = data.target
        Embed = self.Protein_Embed(target)
        conv1 = self.Protein_Conv1(Embed)
        gate1 = torch.sigmoid(self.Protein_Gate1(Embed))
        GCNN1_Output = conv1 * gate1
        GCNN1_Output = self.relu(GCNN1_Output)
        conv2 = self.Protein_Conv2(GCNN1_Output)
        gate2 = torch.sigmoid(self.Protein_Gate2(GCNN1_Output))
        GCNN2_Output = conv2 * gate2
        GCNN2_Output = self.relu(GCNN2_Output)
        conv3 = self.Protein_Conv3(GCNN2_Output)
        gate3 = torch.sigmoid(self.Protein_Gate3(GCNN2_Output))
        GCNN3_Output = conv3 * gate3
        GCNN3_Output = self.relu(GCNN3_Output)
        xt = GCNN3_Output.view(-1, 96 * 107)
        xt = self.Protein_FC(xt)
        return xt, GCNN3_Output

class CrossAttentionFusion(nn.Module):

    def __init__(self, dim, num_heads, dropout=0.1):
        super().__init__()
        self.attn_drug2protein = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.attn_protein2drug = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.drug_ffn = nn.Sequential(
            nn.Linear(dim, 2 * dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * dim, dim)
        )
        self.protein_ffn = nn.Sequential(
            nn.Linear(dim, 2 * dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * dim, dim)
        )
        self.combined_ffn = nn.Sequential(
            nn.Linear(2 * dim, 4 * dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * dim, 2 * dim)
        )

    def forward(self, drug_feat, protein_feat):
        drug_seq = drug_feat.unsqueeze(1)
        protein_seq = protein_feat.unsqueeze(1)
        attn_drug, _ = self.attn_drug2protein(
            query=drug_seq,
            key=protein_seq,
            value=protein_seq
        )
        attn_drug = self.dropout(attn_drug)
        drug_attn_feat = attn_drug.squeeze(1)
        attn_protein, _ = self.attn_protein2drug(
            query=protein_seq,
            key=drug_seq,
            value=drug_seq
        )
        attn_protein = self.dropout(attn_protein)
        protein_attn_feat = attn_protein.squeeze(1)
        drug_ffn_out = self.drug_ffn(drug_attn_feat)
        protein_ffn_out = self.protein_ffn(protein_attn_feat)
        combined = torch.cat([drug_ffn_out, protein_ffn_out], dim=1)
        fused = self.combined_ffn(combined)
        return fused, drug_ffn_out, protein_ffn_out

class SplineLayer(nn.Module):
    def __init__(self, num_inputs, num_splines=20, spline_range=(-5.0, 5.0)):
        super(SplineLayer, self).__init__()
        self.num_inputs = num_inputs
        self.num_splines = num_splines
        self.spline_range = spline_range
        self.register_buffer('knots', torch.linspace(spline_range[0], spline_range[1], num_splines))
        self.weights = nn.Parameter(torch.randn(num_inputs, num_splines) * 0.01)
        self.scale = nn.Parameter(torch.ones(num_inputs) * 0.1)
        
    def forward(self, x):
        B, D = x.shape
        x_expanded = x.unsqueeze(-1)
        knots = self.knots.to(x.device)
        dist = (x_expanded - knots) / (knots[1] - knots[0]) * self.scale.unsqueeze(-1)
        basis = torch.zeros_like(dist)
        mask1 = torch.abs(dist) < 1
        basis[mask1] = 2/3 - torch.pow(torch.abs(dist[mask1]), 2) + 0.5 * torch.pow(torch.abs(dist[mask1]), 3)
        mask2 = (torch.abs(dist) >= 1) & (torch.abs(dist) < 2)
        basis[mask2] = torch.pow(2 - torch.abs(dist[mask2]), 3) / 6
        out = torch.sum(basis * self.weights.unsqueeze(0), dim=-1)
        return out

class FC(torch.nn.Module):
    def __init__(self, input_dim, n_output, dropout):
        super(FC, self).__init__()
        self.FC_layers = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, n_output)
        )

    def forward(self, fused_feat):
        return self.FC_layers(fused_feat)

class LST_Net(nn.Module):

    def __init__(self, input_dim, n_output, dropout=0.1):
        super(LST_Net, self).__init__()
        self.norm = nn.BatchNorm1d(input_dim)
        self.feature_gate = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Sigmoid()
        )
        self.spline = SplineLayer(
            num_inputs=input_dim,
            num_splines=30,
            spline_range=(-5.0, 5.0)
        )
        self.fc_layers = FC(input_dim, n_output, dropout)
        
    def forward(self, x):
        x_norm = self.norm(x)
        gate = self.feature_gate(x_norm)
        x_filtered = x_norm * gate
        x_spline = self.spline(x_filtered)
        out = self.fc_layers(x_spline)
        return out

class AFE_DTA(torch.nn.Module):

    def __init__(self, tokenizer):
        super(AFE_DTA, self).__init__()
        self.hidden_dim = 376
        self.max_len = 128
        self.node_feature = 94
        self.output_dim = 128
        self.ff_dim = 1024
        self.heads = 8
        self.layers = 8
        self.encoder_dropout = 0.2
        self.dropout = 0.3
        self.protein_f = 25
        self.filters = 32
        self.kernel = 8
        self.encoder = Encoder(Drug_Features=self.node_feature, dropout=self.encoder_dropout, Final_dim=self.output_dim)
        self.decoder = Decoder(dim=self.hidden_dim, ff_dim=self.ff_dim, num_head=self.heads, num_layer=self.layers)
        self.dencoder = TransformerEncoder(dim=self.hidden_dim, ff_dim=self.ff_dim, num_head=self.heads, num_layer=self.layers)
        self.pos_encoding = PositionalEncoding(self.hidden_dim, max_len=139)
        self.cnn = GatedCNN(Protein_Features=self.protein_f, Num_Filters=self.filters, 
                            Embed_dim=self.output_dim, Final_dim=self.output_dim, K_size=self.kernel)
        self.cross_attn_fusion = CrossAttentionFusion(
            dim=self.output_dim,
            num_heads=self.heads,
            dropout=self.dropout
        )
        self.fc = LST_Net(input_dim=4 * self.output_dim, n_output=1, dropout=self.dropout)
        self.zz_seg_encoding = nn.Parameter(torch.randn(self.hidden_dim))
        vocab_size = len(tokenizer)
        self.word_pred = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.PReLU(),
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, vocab_size)
        )
        torch.nn.init.zeros_(self.word_pred[3].bias)
        self.vocab_size = vocab_size
        self.sos_value = tokenizer.s2i['<sos>']
        self.eos_value = tokenizer.s2i['<eos>']
        self.pad_value = tokenizer.s2i['<pad>']
        self.word_embed = nn.Embedding(vocab_size, self.hidden_dim)
        self.unk_index = Tokenizer.SPECIAL_TOKENS.index('<unk>')
        self.expand = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            Rearrange('batch_size h -> 1 batch_size h')
        )

    def expand_then_fusing(self, z, pp_mask, vvs):
        zz = z
        zzs = zz + self.zz_seg_encoding
        full_mask = zz.new_zeros(zz.shape[1], zz.shape[0])
        full_mask = torch.cat((pp_mask, full_mask), dim=1)
        zzz = torch.cat((vvs, zzs), dim=0)
        zzz = self.dencoder(zzz, full_mask)
        return zzz, full_mask

    def sample(self, batch_size, device):
        z = torch.randn(1, self.hidden_dim).to(device)
        return z

    def forward(self, data):

        Protein_vector, con = self.cnn(data)
        vss, AOUT, mask, Drug_feature, kl_loss = self.encoder(data, con)
        fused, drug_ffn_out, protein_ffn_out = self.cross_attn_fusion(Drug_feature, Protein_vector)
        Prediction = self.fc(fused)
        zzz, encoder_mask = self.expand_then_fusing(AOUT, mask, vss)
        targets = data.target_seq
        _, target_length = targets.shape
        target_mask = torch.triu(torch.ones(target_length, target_length, dtype=torch.bool), diagonal=1).to(targets.device)
        target_embed = self.word_embed(targets)
        target_embed = self.pos_encoding(target_embed.permute(1, 0, 2).contiguous())
        output = self.decoder(target_embed, zzz, x_mask=target_mask, mem_padding_mask=encoder_mask)
        output = output.permute(1, 0, 2).contiguous()
        prediction_scores = self.word_pred(output)
        shifted_prediction_scores = prediction_scores[:, :-1, :].contiguous()
        targets = targets[:, 1:].contiguous()
        batch_size, sequence_length, vocab_size = shifted_prediction_scores.size()
        shifted_prediction_scores = shifted_prediction_scores.view(-1, vocab_size)
        targets = targets.view(-1)
        lm_loss = F.cross_entropy(shifted_prediction_scores, targets, ignore_index=self.pad_value)
        return Prediction, prediction_scores, lm_loss, kl_loss

    def _generate(self, zzz, encoder_mask, random_sample, return_score=False):
        batch_size = zzz.shape[1]
        device = zzz.device
        token = torch.full((batch_size, self.max_len), self.pad_value, dtype=torch.long, device=device)
        token[:, 0] = self.sos_value
        text_pos = self.pos_encoding.pe
        text_embed = self.word_embed(token[:, 0])
        text_embed = text_embed + text_pos[0]
        text_embed = text_embed.unsqueeze(0)
        incremental_state = torch.jit.annotate(
            Dict[str, Dict[str, Optional[torch.Tensor]]],
            torch.jit.annotate(Dict[str, Dict[str, Optional[torch.Tensor]]], {}),
        )
        if return_score:
            scores = []
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        for t in range(1, self.max_len):
            one = self.decoder.forward_one(text_embed, zzz, incremental_state, mem_padding_mask=encoder_mask)
            one = one.squeeze(0)
            l = self.word_pred(one)
            if return_score:
                scores.append(l)
            if random_sample:
                k = torch.multinomial(torch.softmax(l, 1), 1).squeeze(1)
            else:
                k = torch.argmax(l, -1)
            token[:, t] = k
            finished |= k == self.eos_value
            if finished.all():
                break
            text_embed = self.word_embed(k)
            text_embed = text_embed + text_pos[t]
            text_embed = text_embed.unsqueeze(0)
        predict = token[:, 1:]
        if return_score:
            return predict, torch.stack(scores, dim=1)
        return predict

    def generate(self, data, random_sample=False, return_z=False):
        _, con = self.cnn(data)
        vss, AOUT, mask, Drug_feature, kl_loss = self.encoder(data, con)
        z = self.sample(data.batch, device=vss.device)
        zzz, encoder_mask = self.expand_then_fusing(AOUT, mask, vss)
        predict = self._generate(zzz, encoder_mask, random_sample=random_sample, return_score=False)
        if return_z:
            return predict, z.detach().cpu().numpy()
        return predict
