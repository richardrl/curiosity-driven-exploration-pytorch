import torch.nn.functional as F
import torch.nn as nn
import torch
import torch.optim as optim
import numpy as np
import math
from torch.nn import init


class NoisyLinear(nn.Module):
    """Factorised Gaussian NoisyNet"""

    def __init__(self, in_features, out_features, sigma0=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.noisy_weight = nn.Parameter(
            torch.Tensor(out_features, in_features))
        self.noisy_bias = nn.Parameter(torch.Tensor(out_features))
        self.noise_std = sigma0 / math.sqrt(self.in_features)

        self.reset_parameters()
        self.register_noise()

    def register_noise(self):
        in_noise = torch.FloatTensor(self.in_features)
        out_noise = torch.FloatTensor(self.out_features)
        noise = torch.FloatTensor(self.out_features, self.in_features)
        self.register_buffer('in_noise', in_noise)
        self.register_buffer('out_noise', out_noise)
        self.register_buffer('noise', noise)

    def sample_noise(self):
        self.in_noise.normal_(0, self.noise_std)
        self.out_noise.normal_(0, self.noise_std)
        self.noise = torch.mm(
            self.out_noise.view(-1, 1), self.in_noise.view(1, -1))

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.noisy_weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
            self.noisy_bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        """
        Note: noise will be updated if x is not volatile
        """
        normal_y = nn.functional.linear(x, self.weight, self.bias)
        if self.training:
            # update the noise once per update
            self.sample_noise()

        noisy_weight = self.noisy_weight * self.noise
        noisy_bias = self.noisy_bias * self.out_noise
        noisy_y = nn.functional.linear(x, noisy_weight, noisy_bias)
        return noisy_y + normal_y

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) + ')'


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class ActorCriticNetwork(nn.Module):
    """
    State version, no convolutions
    """
    def __init__(self, input_size, output_size):
        super(ActorCriticNetwork, self).__init__()
        self.actor_fc1 = nn.Linear(input_size, 512)
        self.actor_fc2 = nn.Linear(512, 512)
        self.actor_fc3 = nn.Linear(512, output_size)
        self.leakyReLU = nn.LeakyReLU()

        self.critic_fc1 = nn.Linear(input_size, 512)
        self.critic_fc2 = nn.Linear(512, 512)
        self.critic_fc3 = nn.Linear(512, 1)

    def forward(self, state):
        """

        :param state: (batch, obs_len)
        :return:
        """
        state = state.view(state.shape[0], -1)
        x = self.actor_fc1(state)
        x = self.leakyReLU(x)
        x = self.actor_fc2(x)
        x = self.leakyReLU(x)
        policy = self.actor_fc3(x)

        x = self.critic_fc1(state)
        x = self.leakyReLU(x)
        x = self.critic_fc2(x)
        x = self.leakyReLU(x)
        value = self.critic_fc3(x)
        return policy, value

class CnnActorCriticNetwork(nn.Module):
    def __init__(self, input_size, output_size, use_noisy_net=False):
        super(CnnActorCriticNetwork, self).__init__()

        if use_noisy_net:
            print('use NoisyNet')
            linear = NoisyLinear
        else:
            linear = nn.Linear

        self.feature = nn.Sequential(
            nn.Conv2d(
                in_channels=4,
                out_channels=32,
                kernel_size=8,
                stride=4),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=4,
                stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1),
            nn.LeakyReLU(),
            Flatten(),
            linear(
                # 7 * 7 * 64, changing for 84 -> 42
                64,
                512),
            nn.LeakyReLU()
        )

        self.actor = nn.Sequential(
            linear(512, 512),
            nn.LeakyReLU(),
            linear(512, output_size)
        )

        self.critic = nn.Sequential(
            linear(512, 512),
            nn.LeakyReLU(),
            linear(512, 1)
        )

        for p in self.modules():
            if isinstance(p, nn.Conv2d):
                init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()

            if isinstance(p, nn.Linear):
                init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()

        for i in range(len(self.actor)):
            if type(self.actor[i]) == nn.Linear:
                init.orthogonal_(self.actor[i].weight, 0.01)
                self.actor[i].bias.data.zero_()

        for i in range(len(self.critic)):
            if type(self.critic[i]) == nn.Linear:
                init.orthogonal_(self.critic[i].weight, 0.01)
                self.critic[i].bias.data.zero_()

    def forward(self, state):
        x = self.feature(state)
        policy = self.actor(x)
        value = self.critic(x)
        return policy, value

class ICMModelState(nn.Module):
    def __init__(self, input_size, output_size,):
        super(ICMModelState, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.feature = nn.Linear(input_size * 4, 512)

        self.inverse_net_fc1 = nn.Linear(512 * 2, 512)
        self.inverse_net_fc2 = nn.Linear(512, output_size)

        self.leakyReLU = nn.LeakyReLU()

        self.forward_net_1_fc1 = nn.Linear(output_size + 512, 512)

        self.residual = [nn.Sequential(
            nn.Linear(output_size + 512, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 512),
        )] * 8

        self.forward_net_2_fc1 = nn.Linear(512 + output_size, 512)

    def forward(self, inputs):
        # (256, 4, 142), (256, 4, 142), (256, 6) so action is onehot? Yes, action one hot is correct
        # (16, 568), (16, 568), (16, 6)
        state, next_state, action = inputs
        assert len(state.shape) == 3 and len(next_state.shape) == 3
        assert state.shape[1] == 4 and state.shape[2] == self.input_size, state.shape
        assert next_state.shape[1] == 4 and next_state.shape[2] == self.input_size, next_state.shape
        state = state.view(state.shape[0], -1)
        next_state = next_state.view(next_state.shape[0], -1)

        state = self.feature(state)
        next_state = self.feature(next_state)
        # get pred action
        # pred_action = torch.cat((encode_state, encode_next_state), 1)
        # (256, 8, 142)
        pred_action = torch.cat((state, next_state), 1)


        x = self.inverse_net_fc1(pred_action)
        pred_action = self.inverse_net_fc2(x)

        # ---------------------

        # get pred next state
        # pred_next_state_feature_orig = torch.cat((encode_state, action), 1)

        pred_next_state_feature_orig = torch.cat((next_state, action), 1)
        x = self.forward_net_1_fc1(pred_next_state_feature_orig)
        pred_next_state_feature_orig = self.leakyReLU(x)

        # residual
        for i in range(4):
            pred_next_state_feature = self.residual[i * 2](torch.cat((pred_next_state_feature_orig, action), 1))
            pred_next_state_feature_orig = self.residual[i * 2 + 1](
                torch.cat((pred_next_state_feature, action), 1)) + pred_next_state_feature_orig

        pred_next_state_feature = self.forward_net_2_fc1(torch.cat((pred_next_state_feature_orig, action), 1))

        real_next_state_feature = next_state
        return real_next_state_feature, pred_next_state_feature, pred_action

class ICMModel(nn.Module):
    def __init__(self, input_size, output_size, use_cuda=True):
        super(ICMModel, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.device = torch.device('cuda' if use_cuda else 'cpu')

        # feature_output = 7 * 7 * 64
        feature_output = 64
        self.feature = nn.Sequential(
            nn.Conv2d(
                in_channels=4,
                out_channels=32,
                kernel_size=8,
                stride=4),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=4,
                stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1),
            nn.LeakyReLU(),
            Flatten(),
            nn.Linear(feature_output, 512)
        )

        self.inverse_net = nn.Sequential(
            nn.Linear(512 * 2, 512),
            nn.ReLU(),
            nn.Linear(512, output_size)
        )

        self.residual = [nn.Sequential(
            nn.Linear(output_size + 512, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 512),
        ).to(self.device)] * 8

        self.forward_net_1 = nn.Sequential(
            nn.Linear(output_size + 512, 512),
            nn.LeakyReLU()
        )
        self.forward_net_2 = nn.Sequential(
            nn.Linear(output_size + 512, 512),
        )

        for p in self.modules():
            if isinstance(p, nn.Conv2d):
                init.kaiming_uniform_(p.weight)
                p.bias.data.zero_()

            if isinstance(p, nn.Linear):
                init.kaiming_uniform_(p.weight, a=1.0)
                p.bias.data.zero_()

    def forward(self, inputs):
        # (16, 4, 42, 42), (16, 4, 42, 42), (16, 6)
        state, next_state, action = inputs

        encode_state = self.feature(state)
        encode_next_state = self.feature(next_state)
        # get pred action
        # (16, 1024)
        pred_action = torch.cat((encode_state, encode_next_state), 1)

        pred_action = self.inverse_net(pred_action)
        # ---------------------

        # get pred next state
        pred_next_state_feature_orig = torch.cat((encode_state, action), 1)
        pred_next_state_feature_orig = self.forward_net_1(pred_next_state_feature_orig)

        # residual
        for i in range(4):
            pred_next_state_feature = self.residual[i * 2](torch.cat((pred_next_state_feature_orig, action), 1))
            pred_next_state_feature_orig = self.residual[i * 2 + 1](
                torch.cat((pred_next_state_feature, action), 1)) + pred_next_state_feature_orig

        pred_next_state_feature = self.forward_net_2(torch.cat((pred_next_state_feature_orig, action), 1))

        real_next_state_feature = encode_next_state
        return real_next_state_feature, pred_next_state_feature, pred_action