# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

class Spatial_Attention_layer(nn.Module):
    '''
    compute spatial attention scores
    '''
    def __init__(self, DEVICE, in_channels, num_of_vertices, num_of_timesteps):
        super(Spatial_Attention_layer, self).__init__()
        self.W1 = nn.Parameter(torch.FloatTensor(num_of_timesteps).to(DEVICE)) # (T)
        self.W2 = nn.Parameter(torch.FloatTensor(in_channels, num_of_timesteps).to(DEVICE)) # (F,T)
        self.W3 = nn.Parameter(torch.FloatTensor(in_channels).to(DEVICE)) # F
        self.bs = nn.Parameter(torch.FloatTensor(1, num_of_vertices, num_of_vertices).to(DEVICE)) # (1,N,N)
        self.Vs = nn.Parameter(torch.FloatTensor(num_of_vertices, num_of_vertices).to(DEVICE)) # (N,N)

    def forward(self, x):
        '''
        :param x: (batch_size, N, F_in, T)
        :return: (B,N,N)
        '''

        lhs = torch.matmul(torch.matmul(x, self.W1), self.W2)  # (b,N,F,T)(T)->(b,N,F)(F,T)->(b,N,T)

        rhs = torch.matmul(self.W3, x).transpose(-1, -2)  # (F)(b,N,F,T)->(b,N,T)->(b,T,N)

        product = torch.matmul(lhs, rhs)  # (b,N,T)(b,T,N) -> (B, N, N)

        S = torch.matmul(self.Vs, torch.sigmoid(product + self.bs))  # (N,N)(B, N, N)->(B,N,N)

        S_normalized = F.softmax(S, dim=1)

        return S_normalized

# Adaptive Graph Convolution-AASTBlock
class napl_conv_withSAT(nn.Module):
    '''
       K-order Adaptive Graph Convolution
    '''
    def __init__(self, K, in_channels, out_channels):
        super(napl_conv_withSAT, self).__init__()
        self.K = K
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.Theta = nn.ParameterList([nn.Parameter(torch.FloatTensor(in_channels, out_channels).to('cuda')) for _ in range(K)])

    def forward(self, x, spatial_attention,node_embeddings):
        '''
            napl adaptive graph convolution operation
            :param x: (batch_size, N, F_in, T)
            :return: (batch_size, N, F_out, T)
        '''
        batch_size, num_of_vertices, in_channels, num_of_timesteps = x.shape

        # Adaptive Node Embedding for Et=ErErT
        supports = F.softmax(F.relu(torch.mm(node_embeddings, node_embeddings.transpose(0, 1))), dim=1)
        support_polynomials = [torch.eye(num_of_vertices).to(supports.device), supports] ## [[N_one,N_one],[N,N]]
        for k in range(2, self.K):
            support_polynomials.append(torch.matmul(2 * supports, support_polynomials[-1]) - support_polynomials[-2])

        outputs = []
        for time_step in range(num_of_timesteps):
            graph_signal = x[:, :, :, time_step]  # (b, N, F_in)
            output = torch.zeros(batch_size, num_of_vertices, self.out_channels).to('cuda')  # (b, N, F_out)
            for k in range(self.K):
                T_k = support_polynomials[k]  # (N,N) #
                T_k_with_at = T_k.mul(spatial_attention)  # (N,N)*(N,N) = (N,N)
                theta_k = self.Theta[k]  # (in_channel, out_channel)
                rhs = T_k_with_at.permute(0, 2, 1).matmul(
                    graph_signal)  # (N, N)(b, N, F_in) = (b, N, F_in)  left*
                output = output + rhs.matmul(theta_k)  # (b, N, F_in)(F_in, F_out) = (b, N, F_out)
            outputs.append(output.unsqueeze(-1))  # (b, N, F_out, 1)
        return F.relu(torch.cat(outputs, dim=-1))  # (b, N, F_out, T)

# cut for sparse attention
class Sparse_Spatial_Attention_layer(nn.Module):
    '''
    compute spatial attention scores
    '''
    def __init__(self, DEVICE, in_channels, num_of_vertices, num_of_timesteps):
        super(Sparse_Spatial_Attention_layer, self).__init__()
        self.W1 = nn.Parameter(torch.FloatTensor(num_of_timesteps).to(DEVICE)) # (T)
        self.W2 = nn.Parameter(torch.FloatTensor(in_channels, num_of_timesteps).to(DEVICE)) # (F,T)
        self.W3 = nn.Parameter(torch.FloatTensor(in_channels).to(DEVICE)) # F
        self.bs = nn.Parameter(torch.FloatTensor(1, num_of_vertices, num_of_vertices).to(DEVICE)) # (1,N,N)
        self.Vs = nn.Parameter(torch.FloatTensor(num_of_vertices, num_of_vertices).to(DEVICE)) # (N,N)

    def forward(self, x):
        '''
        :param x: (batch_size, N, F_in, T)
        :return: (B,N,N)
        '''

        lhs = torch.matmul(torch.matmul(x, self.W1), self.W2)  # (b,N,F,T)(T)->(b,N,F)(F,T)->(b,N,T)
        rhs = torch.matmul(self.W3, x).transpose(-1, -2)  # (F)(b,N,F,T)->(b,N,T)->(b,T,N)
        product = torch.matmul(lhs, rhs)  # (b,N,T)(b,T,N) -> (B, N, N)
        S = torch.matmul(self.Vs, torch.sigmoid(product + self.bs))  # (N,N)(B, N, N)->(B,N,N)
        S_normalized = F.softmax(S, dim=1)
        # print('S_normalized:{}'.format(S_normalized))

        # Ablation Study
        # A1: without cut s operation
        # return S_normalized

        # A2: Sparse operation-mean↑
        zero = torch.zeros_like(S)
        # default theta is 0.6 and 0.1 corresponding overall results. 
        theta = 0.6
        Sparse_S_normalized = torch.where(S_normalized < torch.div(torch.mean(S_normalized),theta),zero,S_normalized)
        print('Sparse_S_normalized:')
        print(Sparse_S_normalized)
        # A3: Sparse operation-mean + resoftmax↑↑ best
        Sparse_S_normalized_resoftmax = F.softmax(Sparse_S_normalized,dim=1)
         
        print('Sparse_S_normalized_resoftmax.shape:{}'.format(Sparse_S_normalized_resoftmax.shape))


        # spatial_attention = Sparse_S_normalized_resoftmax[-1].detach().cpu().numpy()
        # np.savetxt('sparse_spatial_attention.csv', spatial_attention,delimiter=',')
        # print('save success')

        print('Sparse_S_normalized_resoftmax:')
        print(Sparse_S_normalized_resoftmax)
        # original best results
        # return Sparse_S_normalized_resoftmax
        return Sparse_S_normalized

        

# original cheb_conv in astgcn
class cheb_conv_withSAt(nn.Module):
    '''
    K-order chebyshev graph convolution
    '''

    def __init__(self, K, cheb_polynomials, in_channels, out_channels):
        '''
        :param K: int
        :param in_channles: int, num of channels in the input sequence
        :param out_channels: int, num of channels in the output sequence
        '''
        super(cheb_conv_withSAt, self).__init__()
        self.K = K
        self.cheb_polynomials = cheb_polynomials
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.DEVICE = cheb_polynomials[0].device
        self.Theta = nn.ParameterList([nn.Parameter(torch.FloatTensor(in_channels, out_channels).to('cuda')) for _ in range(K)])

    def forward(self, x, spatial_attention):
        '''
        Chebyshev graph convolution operation
        :param x: (batch_size, N, F_in, T)
        :return: (batch_size, N, F_out, T)
        '''

        batch_size, num_of_vertices, in_channels, num_of_timesteps = x.shape

        outputs = []

        for time_step in range(num_of_timesteps):

            graph_signal = x[:, :, :, time_step]  # (b, N, F_in)

            output = torch.zeros(batch_size, num_of_vertices, self.out_channels).to('cuda')  # (b, N, F_out)

            for k in range(self.K):

                T_k = self.cheb_polynomials[k]  # (N,N)

                T_k_with_at = T_k.mul(spatial_attention)   # (N,N)*(N,N) = (N,N)

                theta_k = self.Theta[k]  # (in_channel, out_channel)

                rhs = T_k_with_at.permute(0, 2, 1).matmul(graph_signal)  # (N, N)(b, N, F_in) = (b, N, F_in)

                output = output + rhs.matmul(theta_k)  # (b, N, F_in)(F_in, F_out) = (b, N, F_out)

            outputs.append(output.unsqueeze(-1))  # (b, N, F_out, 1)

        return F.relu(torch.cat(outputs, dim=-1))  # (b, N, F_out, T)


class Temporal_Attention_layer(nn.Module):
    """
    num_of_vertices:Nodes num
    in_channels: input_dim
    num_of_timesteps:

    """
    def __init__(self, DEVICE, in_channels, num_of_vertices, num_of_timesteps):
        super(Temporal_Attention_layer, self).__init__()
        self.U1 = nn.Parameter(torch.FloatTensor(num_of_vertices).to(DEVICE)) # N
        self.U2 = nn.Parameter(torch.FloatTensor(in_channels, num_of_vertices).to(DEVICE)) # F_in,N
        self.U3 = nn.Parameter(torch.FloatTensor(in_channels).to(DEVICE)) # F_in
        self.be = nn.Parameter(torch.FloatTensor(1, num_of_timesteps, num_of_timesteps).to(DEVICE)) # 1, T, T
        self.Ve = nn.Parameter(torch.FloatTensor(num_of_timesteps, num_of_timesteps).to(DEVICE)) # T, T

    def forward(self, x):
        '''
        :param x: (batch_size, N, F_in, T)
        :return: (B, T, T)
        '''
        _, num_of_vertices, num_of_features, num_of_timesteps = x.shape # _, N, F_in, T

        lhs = torch.matmul(torch.matmul(x.permute(0, 3, 2, 1), self.U1), self.U2)
        # x:(B, N, F_in, T) -> (B, T, F_in, N)
        # (B, T, F_in, N)(N) -> (B,T,F_in)
        # (B,T,F_in)(F_in,N)->(B,T,N)

        rhs = torch.matmul(self.U3, x)  # (F)(B,N,F,T)->(B, N, T)

        product = torch.matmul(lhs, rhs)  # (B,T,N)(B,N,T)->(B,T,T)

        E = torch.matmul(self.Ve, torch.sigmoid(product + self.be))  # (B, T, T)

        E_normalized = F.softmax(E, dim=1)

        return E_normalized


class cheb_conv(nn.Module):
    '''
    K-order chebyshev graph convolution
    '''

    def __init__(self, K, cheb_polynomials, in_channels, out_channels):
        '''
        :param K: int
        :param in_channles: int, num of channels in the input sequence
        :param out_channels: int, num of channels in the output sequence
        '''
        super(cheb_conv, self).__init__()
        self.K = K
        self.cheb_polynomials = cheb_polynomials
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.DEVICE = cheb_polynomials[0].device
        self.Theta = nn.ParameterList([nn.Parameter(torch.FloatTensor(in_channels, out_channels).to(self.DEVICE)) for _ in range(K)])

    def forward(self, x):
        '''
        Chebyshev graph convolution operation
        :param x: (batch_size, N, F_in, T)
        :return: (batch_size, N, F_out, T)
        '''

        batch_size, num_of_vertices, in_channels, num_of_timesteps = x.shape

        outputs = []

        for time_step in range(num_of_timesteps):

            graph_signal = x[:, :, :, time_step]  # (b, N, F_in)

            output = torch.zeros(batch_size, num_of_vertices, self.out_channels).to(self.DEVICE)  # (b, N, F_out)

            for k in range(self.K):

                T_k = self.cheb_polynomials[k]  # (N,N)

                theta_k = self.Theta[k]  # (in_channel, out_channel)

                rhs = graph_signal.permute(0, 2, 1).matmul(T_k).permute(0, 2, 1)

                output = output + rhs.matmul(theta_k)

            outputs.append(output.unsqueeze(-1))

        return F.relu(torch.cat(outputs, dim=-1))


class AAST_block(nn.Module):
    def __init__(self, DEVICE, in_channels, K, nb_chev_filter, nb_time_filter, time_strides, num_of_vertices, num_of_timesteps):
        super(AAST_block, self).__init__()
        self.TAt = Temporal_Attention_layer(DEVICE, in_channels, num_of_vertices, num_of_timesteps)
        # self.SAt = Spatial_Attention_layer(DEVICE, in_channels, num_of_vertices, num_of_timesteps)
        self.SAt = Sparse_Spatial_Attention_layer(DEVICE, in_channels, num_of_vertices, num_of_timesteps)
        # The adaptive convolution for the spatial-temporal attention mechanism
        self.napl_conv_SAT = napl_conv_withSAT(K, in_channels, nb_chev_filter)
        # original chebs convolution
        # self.cheb_conv_SAt = cheb_conv_withSAt(K, cheb_polynomials, in_channels, nb_chev_filter)
        
        # original time_conv with LN
        # self.time_conv = nn.Conv2d(nb_chev_filter, nb_time_filter, kernel_size=(1, 3), stride=(1, time_strides), padding=(0, 1))
        self.time_conv = nn.utils.weight_norm(nn.Conv2d(nb_chev_filter, nb_time_filter, kernel_size=(1, 3), stride=(1, time_strides), padding=(0, 1)))
        
        self.residual_conv = nn.Conv2d(in_channels, nb_time_filter, kernel_size=(1, 1), stride=(1, time_strides))
        self.ln = nn.LayerNorm(nb_time_filter)

    def forward(self, x, node_embeddings):
        '''
        :param x: (batch_size, N, F_in, T)
        :return: (batch_size, N, nb_time_filter, T)
        '''
        batch_size, num_of_vertices, num_of_features, num_of_timesteps = x.shape

        # oriignal TAt the following 2 lines
        temporal_At = self.TAt(x)  # (b, T, T)
        x_TAt = torch.matmul(x.reshape(batch_size, -1, num_of_timesteps), temporal_At).reshape(batch_size, num_of_vertices, num_of_features, num_of_timesteps) # B,N,F_in,T
        
        # x:(B,?,T)(B,T,T) ->(B,?,T)
        # (B,?,T)-?(B,N,F_in,T) #

        # Ablation Study for Temporal Attention Start
        # print('x.shape:{}'.format(x.shape)) # B,N,F,T
        # print('X_TAt.shape:{}'.format(x_TAt.shape)) # B,N,F,T
        # x_TAt = x
        # Ablation Study for Temporal Attention End

        # SAt
        spatial_At = self.SAt(x_TAt)

        # cheb gcn
        # spatial_gcn = self.cheb_conv_SAt(x, spatial_At)  # (b,N,F,T)

        # cheb gcn without SAT
        # spatial_gcn = self.cheb_conv(x)

        # Adaptive gcn
        spatial_gcn = self.napl_conv_SAT(x, spatial_At,node_embeddings)
        # print('spatial_gcn.shape:{}'.format(spatial_gcn.shape))

         
        # Ablation Study for TaCC
        # temp = spatial_gcn.permute(0,2,1,3)
        # x_residual = self.ln(temp.permute(0, 3, 2, 1)).permute(0, 2, 3, 1)

        
        # original TaCC
        # convolution along the time axis
        time_conv_output = self.time_conv(spatial_gcn.permute(0, 2, 1, 3))  # (b,N,F,T)->(b,F,N,T) 用(1,3)的卷积核去做->(b,F,N,T)

        # residual shortcut
        x_residual = self.residual_conv(x.permute(0, 2, 1, 3))  # (b,N,F,T)->(b,F,N,T) 用(1,1)的卷积核去做->(b,F,N,T)
        # original use relu as activation function
        # x_residual = self.ln(F.relu(x_residual + time_conv_output).permute(0, 3, 2, 1)).permute(0, 2, 3, 1)
        # (b,F,N,T)->(b,T,N,F) -ln-> (b,T,N,F)->(b,N,F,T)

        x_residual = self.ln((x_residual + time_conv_output).permute(0, 3, 2, 1)).permute(0, 2, 3, 1)

        
        return x_residual, spatial_At


class ASTGCN_framework(nn.Module):

    def __init__(self, DEVICE, nb_block, in_channels, K, nb_chev_filter, nb_time_filter, time_strides, num_for_predict, len_input, num_of_vertices):
        '''
        :param nb_block:
        :param in_channels:
        :param K:
        :param nb_chev_filter:
        :param nb_time_filter:
        :param time_strides:
        :param cheb_polynomials:
        :param nb_predict_step:
        '''

        super(ASTGCN_framework, self).__init__()
        self.num_node = num_of_vertices
        self.embed_dim = 100
        self.node_embeddings = nn.Parameter(torch.randn(self.num_node, self.embed_dim), requires_grad=True) # N,d

        self.BlockList = nn.ModuleList([AAST_block(DEVICE, in_channels, K, nb_chev_filter, nb_time_filter, time_strides, num_of_vertices, len_input)])

        self.BlockList.extend([AAST_block(DEVICE, nb_time_filter, K, nb_chev_filter, nb_time_filter, 1, num_of_vertices, len_input // time_strides) for _ in range(nb_block - 1)])

        self.final_conv = nn.Conv2d(int(len_input/time_strides), num_for_predict, kernel_size=(1, nb_time_filter))

        self.DEVICE = DEVICE

        self.to(DEVICE)

    def forward(self, x):
        '''
        :param x: (B, N_nodes, F_in, T_in)
        :return: (B, N_nodes, T_out)
        '''
        for block in self.BlockList:
            x = block(x, self.node_embeddings)

        output = self.final_conv(x.permute(0, 3, 1, 2))[:, :, :, -1].permute(0, 2, 1)
        # (b,N,F,T)->(b,T,N,F)-conv<1,F>->(b,c_out*T,N,1)->(b,c_out*T,N)->(b,N,T)

        return output

class ASTGCN_framework_classification(nn.Module):

    def __init__(self, DEVICE, nb_block, in_channels, K, nb_chev_filter, nb_time_filter, time_strides, num_for_predict, len_input, num_of_vertices):
        '''
        :param nb_block:
        :param in_channels:
        :param K:
        :param nb_chev_filter:
        :param nb_time_filter:
        :param time_strides:
        :param cheb_polynomials:
        :param nb_predict_step:
        '''

        super(ASTGCN_framework_classification, self).__init__()
        self.num_node = num_of_vertices
        # for ACL18
        # default best is 100
        self.embed_dim = 100
        # for KDD17
        # default best is 30
        # self.embed_dim = 30
        # self.embed_dim = 6
        # 2022-07-31 try to change torch.randn to torch.zeros to keep stability
        self.node_embeddings = nn.Parameter(torch.randn(self.num_node, self.embed_dim), requires_grad=True) # N,d
        # self.node_embeddings = nn.Parameter(torch.zeros(self.num_node, self.embed_dim), requires_grad=True) # N,d

        self.BlockList = nn.ModuleList([AAST_block(DEVICE, in_channels, K, nb_chev_filter, nb_time_filter, time_strides, num_of_vertices, len_input)])

        self.BlockList.extend([AAST_block(DEVICE, nb_time_filter, K, nb_chev_filter, nb_time_filter, 1, num_of_vertices, len_input // time_strides) for _ in range(nb_block - 1)])

        self.final_conv = nn.Conv2d(int(len_input/time_strides), num_for_predict, kernel_size=(1, nb_time_filter))

        # self.linear = torch.nn.Linear(num_for_predict,1)
        # self.sigmoid = torch.nn.Sigmoid()
        self.fc1 = torch.nn.Linear(num_for_predict,2)
        self.DEVICE = DEVICE

        self.to(DEVICE)

    def forward(self, x):
        '''
        :param x: (B, N_nodes, F_in, T_in)
        :return: (B, N_nodes, T_out)
        '''
        for block in self.BlockList:
            x, sat = block(x, self.node_embeddings)

        output = self.final_conv(x.permute(0, 3, 1, 2))[:, :, :, -1].permute(0, 2, 1)
        # (b,N,F,T)->(b,T,N,F)-conv<1,F>->(b,c_out*T,N,1)->(b,c_out*T,N)->(b,N,T)
        output = self.fc1(output)
        # print('output.shape:{}'.format(output.shape))
        
        # original method
        output = F.softmax(output,dim=-1)

        return output,sat

def make_model(DEVICE, nb_block, in_channels, K, nb_chev_filter, nb_time_filter, time_strides, num_for_predict, len_input, num_of_vertices):
    '''

    :param DEVICE:
    :param nb_block:
    :param in_channels:
    :param K:
    :param nb_chev_filter:
    :param nb_time_filter:
    :param time_strides:
    :param cheb_polynomials:
    :param nb_predict_step:
    :param len_input
    :return:
    '''
    # L_tilde = scaled_Laplacian(adj_mx) # not need
    # cheb_polynomials = [torch.from_numpy(i).type(torch.FloatTensor).to(DEVICE) for i in cheb_polynomial(L_tilde, K)] # replace it with NAPL!
    # remove cheb_polynomials and replace it with NAPL in the forward method
    ### model = ASTGCN_framework(DEVICE, nb_block, in_channels, K, nb_chev_filter, nb_time_filter, time_strides, num_for_predict, len_input, num_of_vertices)
    model = ASTGCN_framework_classification(DEVICE, nb_block, in_channels, K, nb_chev_filter, nb_time_filter, time_strides, num_for_predict, len_input, num_of_vertices)

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.uniform_(p)

    return model
