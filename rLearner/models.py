import math

import torch
import torch.nn as nn
from torch.autograd import Variable


class Movement(nn.Module):
    def __init__(self):
        super(Movement, self).__init__()
        self.bias = nn.Parameter(torch.zeros(1, 1), requires_grad=True)

    def forward(self, x):
        return x + self.bias


def Binarize(tensor, quant_mode='det'):
    if quant_mode == 'det':
        # return tensor.sign()
        input_copy = tensor.clone()
        input_copy[tensor.ge(-100)] = 0
        input_copy[tensor.le(-0.005)] = -1
        input_copy[tensor.ge(0.005)] = 1
        return input_copy
    else:
        return tensor.add_(1).div_(2).add_(torch.rand(tensor.size()).add(-0.5)).clamp_(0, 1).round().mul_(2).add_(-1)


class BinaryActive(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        ctx.save_for_backward(i)
        # input = input.sign()
        input_copy = i.clone()
        input_copy[i.ge(0)] = 1
        input_copy[i.le(0)] = 0
        return input_copy

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input.ge(0.8)] = 0
        grad_input[input.le(-0.8)] = 0
        return grad_input


class BinaryLinear(nn.Linear):
    def __init__(self, *kargs, **kwargs):
        super(BinaryLinear, self).__init__(*kargs, **kwargs)
        self.move = Movement()

    def forward(self, input):
        if not hasattr(self.weight, 'org'):
            self.weight.org = self.weight.data.clone()
        self.weight.data = Binarize(self.weight.org)
        out = nn.functional.linear(self.move(input), self.weight)
        return out


class BinaryGRUCellModify(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, bias=False):
        super(BinaryGRUCellModify, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.x2h = BinaryLinear(input_size, 2 * hidden_size, bias=bias)
        self.move1 = Movement()
        self.fc = BinaryLinear(hidden_size, output_size, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            if w.data.size(0) != 1:
                w.data.uniform_(-std, std)

    def forward(self, x, S1_ori, S2_ori):
        # print('x.size:', x.size())
        x = x.view(-1, x.size(0))
        gate_x = self.x2h(x)
        # gate_x = gate_x.squeeze()
        m1, m2 = gate_x.chunk(2, 1)
        m1_act = nn.Hardtanh()(m1)
        m2_act = nn.Hardtanh()(m2)
        S1_new = S1_ori + m1_act
        S2_new = S2_ori + m2_act
        F_u = BinaryActive.apply(self.move1(S1_new))
        hy = (1 - F_u) * S2_new + F_u * S2_ori
        out = self.fc(hy)
        out = self.softmax(out)
        return S1_new, S2_new, hy, out


class BinaryGRUModelModify(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, use_cpu=True, bias=False):
        super(BinaryGRUModelModify, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.output_dim = output_dim
        self.use_cpu = use_cpu
        self.gru_cells = [BinaryGRUCellModify(input_dim, hidden_dim, output_dim) for i in range(layer_dim)]

    def forward(self, x):
        # x: (x_data, x_length)
        # print(x_data.shape,"x_data.shape")100, 28, 28
        x_data = x[0]
        x_length = x[1]
        x_label = x[2]

        if self.training:
            outs = torch.zeros(x_data.size(0), self.output_dim)
            loss = 0
            for i in range(x_data.size(0)):
                loss_single = 0
                if torch.cuda.is_available() and not self.use_cpu:
                    S1_ori = Variable(torch.zeros(self.layer_dim, 1, self.hidden_dim).cuda())
                    S2_ori = Variable(torch.zeros(self.layer_dim, 1, self.hidden_dim).cuda())
                else:
                    S1_ori = Variable(torch.zeros(self.layer_dim, 1, self.hidden_dim))
                    S2_ori = Variable(torch.zeros(self.layer_dim, 1, self.hidden_dim))
                for seq in range(x_length[i]):
                    for layer in range(self.layer_dim):
                        if layer == 0:
                            S1_ori[layer, :, :], S2_ori[layer, :, :], hn, out = self.gru_cells[layer](x_data[i, seq, :],
                                                                                                      S1_ori[layer, :, :].clone(),
                                                                                                      S2_ori[layer, :, :].clone())
                        else:
                            S1_ori[layer, :, :], S2_ori[layer, :, :], hn, out = self.gru_cells[layer](hn,
                                                                                                      S1_ori[layer, :, :].clone(),
                                                                                                      S2_ori[layer, :, :].clone())
                #     outs[i] += out
                # return torch.sigmoid(outs[:, 1])
                    loss_single += (x_label[i] - torch.sigmoid(out[0, 1]))**2
                loss += loss_single
            return loss
        else:
            predicted = torch.zeros(x_data.size(0))
            for i in range(x_data.size(0)):
                if torch.cuda.is_available() and not self.use_cpu:
                    S1_ori = Variable(torch.zeros(self.layer_dim, 1, self.hidden_dim).cuda())
                    S2_ori = Variable(torch.zeros(self.layer_dim, 1, self.hidden_dim).cuda())
                else:
                    S1_ori = Variable(torch.zeros(self.layer_dim, 1, self.hidden_dim))
                    S2_ori = Variable(torch.zeros(self.layer_dim, 1, self.hidden_dim))
                for seq in range(x_length[i]):
                    for layer in range(self.layer_dim):
                        if layer == 0:
                            S1_ori[layer, :, :], S2_ori[layer, :, :], hn, out = self.gru_cells[layer](x_data[i, seq, :],
                                                                                                      S1_ori[layer, :, :].clone(),
                                                                                                      S2_ori[layer, :, :].clone())
                        else:
                            S1_ori[layer, :, :], S2_ori[layer, :, :], hn, out = self.gru_cells[layer](hn,
                                                                                                      S1_ori[layer, :, :].clone(),
                                                                                                      S2_ori[layer, :, :].clone())
                    if out[0, 1] > 0.75:
                        break
                predicted[i] = out[0, 1]
            return predicted


class GRUCellModify(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, bias=False):
        super(GRUCellModify, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.x2h = nn.Linear(input_size, 2 * hidden_size, bias=bias)
        self.fc = nn.Linear(hidden_size, output_size, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            if w.data.size(0) != 1:
                w.data.uniform_(-std, std)

    def forward(self, x, S1_ori, S2_ori):
        # print('x.size in gru cell:', x.size())
        x = Variable(x.view(-1, x.size(0)).cuda())
        print(x.device)
        print('2222')
        gate_x = self.x2h(x)
        # gate_x = gate_x.squeeze()
        # print('x.size in gru cell:', x.size())
        # print('gate_x.size:', gate_x.size())
        m1, m2 = gate_x.chunk(2, 1)
        m1_act = self.tanh(m1)
        m2_act = self.tanh(m2)
        S1_new = S1_ori + m1_act
        S2_new = S2_ori + m2_act
        F_u = self.relu(S1_new)
        hy = (1 - F_u) * S2_new + F_u * S2_ori
        out = self.fc(hy)
        out = self.softmax(out)
        return S1_new, S2_new, hy, out


class GRUModelModify(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, use_cpu=True, bias=False):
        super(GRUModelModify, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.output_dim = output_dim
        self.gru_cells = [GRUCellModify(input_dim, hidden_dim, output_dim) for i in range(layer_dim)]
        self.use_cpu = use_cpu

    def forward(self, x):
        # x: (x_data, x_length)
        # print(x_data.shape,"x_data.shape")100, 28, 28
        x_data = Variable(x[0].cuda())
        x_length = x[1]
        x_label = x[2]
        if self.training:
            outs = torch.zeros(x_data.size(0), self.output_dim)
            loss = 0
            for i in range(x_data.size(0)):
                loss_single = 0
                if torch.cuda.is_available() and not self.use_cpu:
                    S1_ori = Variable(torch.zeros(self.layer_dim, 1, self.hidden_dim).cuda())
                    S2_ori = Variable(torch.zeros(self.layer_dim, 1, self.hidden_dim).cuda())
                    print('11111')
                else:
                    S1_ori = Variable(torch.zeros(self.layer_dim, 1, self.hidden_dim))
                    S2_ori = Variable(torch.zeros(self.layer_dim, 1, self.hidden_dim))
                    print('22222')
                for seq in range(x_length[i]):
                    for layer in range(self.layer_dim):
                        # print('seq layer S1_ori.size S2_ori.size:', seq, layer, S1_ori.size(), S2_ori.size())
                        if layer == 0:
                            S1_ori[layer, :, :], S2_ori[layer, :, :], hn, out = self.gru_cells[layer](x_data[i, seq, :],
                                                                                                      S1_ori[layer, :, :].clone(),
                                                                                                      S2_ori[layer, :, :].clone())
                        else:
                            S1_ori[layer, :, :], S2_ori[layer, :, :], hn, out = self.gru_cells[layer](hn,
                                                                                                      S1_ori[layer, :, :].clone(),
                                                                                                      S2_ori[layer, :, :].clone())
                    # print('outs.size:', outs.size())
                    # print('outs[i]:', outs[i].size())
                    # print('out.size:', out.size())
                    # outs[i] += out.squeeze()
                    # return torch.sigmoid(outs[:, 1])
                    loss_single += (x_label[i] - torch.sigmoid(out[0, 1]))**2
                loss += loss_single
            return loss
        else:
            predicted = torch.zeros(x_data.size(0))
            for i in range(x_data.size(0)):
                if torch.cuda.is_available() and not self.use_cpu:
                    S1_ori = Variable(torch.zeros(self.layer_dim, 1, self.hidden_dim).cuda())
                    S2_ori = Variable(torch.zeros(self.layer_dim, 1, self.hidden_dim).cuda())
                else:
                    S1_ori = Variable(torch.zeros(self.layer_dim, 1, self.hidden_dim))
                    S2_ori = Variable(torch.zeros(self.layer_dim, 1, self.hidden_dim))
                for seq in range(x_length[i]):
                    for layer in range(self.layer_dim):
                        if layer == 0:
                            S1_ori[layer, :, :], S2_ori[layer, :, :], hn, out = self.gru_cells[layer](x_data[i, seq, :],
                                                                                                      S1_ori[layer, :, :].clone(),
                                                                                                      S2_ori[layer, :, :].clone())
                        else:
                            S1_ori[layer, :, :], S2_ori[layer, :, :], hn, out = self.gru_cells[layer](hn,
                                                                                                      S1_ori[layer, :, :].clone(),
                                                                                                      S2_ori[layer, :, :].clone())
                    # print('out in eval:', out)
                    if out[0, 1] > 0.75:
                        break
                predicted[i] = out[0, 1]
            return predicted
