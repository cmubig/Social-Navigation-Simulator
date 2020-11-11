import torch
import torch.nn as nn


def make_mlp(dim_list, activation='relu', batch_norm=True, dropout=0):
    layers = []
    for dim_in, dim_out in zip(dim_list[:-1], dim_list[1:]):
        layers.append(nn.Linear(dim_in, dim_out))
        if batch_norm:
            layers.append(nn.BatchNorm1d(dim_out))
        if activation == 'relu':
            layers.append(nn.ReLU())
        elif activation == 'leakyrelu':
            layers.append(nn.LeakyReLU())
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
    return nn.Sequential(*layers)


def get_noise(shape, noise_type, aux_input=None):
    if noise_type == 'gaussian':
        return torch.randn(*shape).cuda()
    elif noise_type == 'uniform':
        return torch.rand(*shape).sub_(0.5).mul_(2.0).cuda()
    elif noise_type == 'inject_goal':
        # Specify 'noise_mix_type' to 'individual' to enable goal injection
        return aux_input.view(shape).cuda()

        
    raise ValueError('Unrecognized noise type "%s"' % noise_type)



class Encoder(nn.Module):
    """Encoder is part of both TrajectoryGenerator and
    TrajectoryDiscriminator"""
    def __init__(
        self, embedding_dim=64, h_dim=64, mlp_dim=1024, num_layers=1,
        dropout=0.0
    ):
        super(Encoder, self).__init__()

        self.mlp_dim = 1024
        self.h_dim = h_dim
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        self.encoder = nn.LSTM(
            embedding_dim, h_dim, num_layers, dropout=dropout
        ).cuda()

        self.spatial_embedding = nn.Linear(2, embedding_dim).cuda()

    def init_hidden(self, batch):
        return (
            torch.zeros(self.num_layers, batch, self.h_dim).cuda(),
            torch.zeros(self.num_layers, batch, self.h_dim).cuda()
        )

    def forward(self, obs_traj):
        """
        Inputs:
        - obs_traj: Tensor of shape (obs_len, batch, 2)
        Output:
        - final_h: Tensor of shape (self.num_layers, batch, self.h_dim)
        """
        # Encode observed Trajectory
        batch = obs_traj.size(1)
        obs_traj_embedding = self.spatial_embedding(obs_traj.view(-1, 2)) #(batch*seq) x 2
        obs_traj_embedding = obs_traj_embedding.view(
            -1, batch, self.embedding_dim
        ) # seq x batch x 64
        state_tuple = self.init_hidden(batch)
        output, state = self.encoder(obs_traj_embedding, state_tuple)
        final_h = state[0] #1*batch*64
        return final_h


class Decoder(nn.Module):
    """Decoder is part of TrajectoryGenerator"""
    def __init__(
        self, seq_len, embedding_dim=64, h_dim=128, mlp_dim=1024, num_layers=1,
        pool_every_timestep=True, dropout=0.0, bottleneck_dim=1024,
        activation='relu', batch_norm=True, pooling_type='pool_net',
        neighborhood_size=2.0, grid_size=8, spatial_dim=2
    ):
        super(Decoder, self).__init__()

        self.spatial_dim = spatial_dim
        self.seq_len = seq_len
        self.mlp_dim = mlp_dim
        self.h_dim = h_dim
        self.embedding_dim = embedding_dim
        self.pool_every_timestep = pool_every_timestep

        self.decoder = nn.LSTM(
            embedding_dim, h_dim, num_layers, dropout=dropout
        )

        if pool_every_timestep:
            if pooling_type == 'pool_net':
                self.pool_net = PoolHiddenNet(
                    embedding_dim=self.embedding_dim,
                    h_dim=self.h_dim,
                    mlp_dim=mlp_dim,
                    bottleneck_dim=bottleneck_dim,
                    activation=activation,
                    batch_norm=batch_norm,
                    dropout=dropout
                )
            elif pooling_type == 'spool':
                self.pool_net = SocialPooling(
                    h_dim=self.h_dim,
                    activation=activation,
                    batch_norm=batch_norm,
                    dropout=dropout,
                    neighborhood_size=neighborhood_size,
                    grid_size=grid_size
                )

            mlp_dims = [h_dim + bottleneck_dim, mlp_dim, h_dim]
            self.mlp = make_mlp(
                mlp_dims,
                activation=activation,
                batch_norm=batch_norm,
                dropout=dropout
            )
        self.spatial_embedding = nn.Linear(self.spatial_dim, embedding_dim)
        self.hidden2pos = nn.Linear(h_dim, 2)

    def get_hidden_sequence(self):
        """
        Export sequence of state tuple for further analysis
        state_tuple = (hidden_state, cell_state)
        """
        return self.state_seq

    def forward(self, last_pos, last_pos_rel, state_tuple, seq_start_end, seq_len=8):
        """
        Inputs:
        - last_pos: Tensor of shape (batch, 2)
        - last_pos_rel: Tensor of shape (batch, 2)
        - state_tuple: (hh, ch) each tensor of shape (num_layers, batch, h_dim)
        - seq_start_end: A list of tuples which delimit sequences within batch
        Output:
        - pred_traj: tensor of shape (self.seq_len, batch, 2)
        """
        batch = last_pos.size(0)
        pred_traj_fake_rel = []
        decoder_input = self.spatial_embedding(last_pos_rel)
        decoder_input = decoder_input.view(1, batch, self.embedding_dim)
        self.state_seq = []

        for _ in range(seq_len):
            output, state_tuple = self.decoder(decoder_input, state_tuple)
            rel_pos = self.hidden2pos(output.view(-1, self.h_dim))
            curr_pos = rel_pos + last_pos

            if self.pool_every_timestep:
                decoder_h = state_tuple[0]
                pool_h = self.pool_net(decoder_h, seq_start_end, curr_pos)
                decoder_h = torch.cat(
                    [decoder_h.view(-1, self.h_dim), pool_h], dim=1)
                decoder_h = self.mlp(decoder_h)
                decoder_h = torch.unsqueeze(decoder_h, 0)
                state_tuple = (decoder_h, state_tuple[1])

            embedding_input = rel_pos

            decoder_input = self.spatial_embedding(embedding_input)
            decoder_input = decoder_input.view(1, batch, self.embedding_dim)
            pred_traj_fake_rel.append(rel_pos.view(batch, -1))
            last_pos = curr_pos
            self.state_seq.append(state_tuple)

        pred_traj_fake_rel = torch.stack(pred_traj_fake_rel, dim=0)
        return pred_traj_fake_rel, state_tuple[0]

    def step_forward(self, last_pos, last_pos_rel, state_tuple, seq_start_end):
        """
        Inputs:
        - last_pos: Tensor of shape (batch, 2)
        - last_pos_rel: Tensor of shape (batch, 2)
        - state_tuple: (hh, ch) each tensor of shape (num_layers, batch, h_dim)
        - seq_start_end: A list of tuples which delimit sequences within batch
        Output:
        - pred_pos: tensor of shape (1, batch, 2)
        """
        batch = last_pos.size(0)
        decoder_input = self.spatial_embedding(last_pos_rel)
        decoder_input = decoder_input.view(1, batch, self.embedding_dim)
        self.state_seq = []

        output, state_tuple = self.decoder(decoder_input, state_tuple)
        rel_pos = self.hidden2pos(output.view(-1, self.h_dim))
        curr_pos = rel_pos + last_pos

        if self.pool_every_timestep:
            decoder_h = state_tuple[0]
            pool_h = self.pool_net(decoder_h, seq_start_end, curr_pos)
            decoder_h = torch.cat(
                [decoder_h.view(-1, self.h_dim), pool_h], dim=1)
            decoder_h = self.mlp(decoder_h)
            decoder_h = torch.unsqueeze(decoder_h, 0)
            state_tuple = (decoder_h, state_tuple[1])

        return rel_pos, state_tuple


class PoolHiddenNet(nn.Module):
    """Pooling module as proposed in our paper"""
    def __init__(
        self, embedding_dim=64, h_dim=64, mlp_dim=1024, bottleneck_dim=1024,
        activation='relu', batch_norm=True, dropout=0.0, group_pooling=False
    ):
        super(PoolHiddenNet, self).__init__()

        self.mlp_dim = 1024
        self.h_dim = h_dim
        self.bottleneck_dim = bottleneck_dim
        self.embedding_dim = embedding_dim

        mlp_pre_dim = embedding_dim + h_dim
        mlp_pre_pool_dims = [mlp_pre_dim, 512, bottleneck_dim] #48 x 512x 32

        self.spatial_embedding = nn.Linear(2, embedding_dim)
        self.mlp_pre_pool = make_mlp(
            mlp_pre_pool_dims,
            activation=activation,
            batch_norm=batch_norm,
            dropout=dropout)
        self.group_pooling = group_pooling

    def repeat(self, tensor, num_reps):
        """
        Inputs:
        -tensor: 2D tensor of any shape
        -num_reps: Number of times to repeat each row
        Outpus:
        -repeat_tensor: Repeat each row such that: R1, R1, R2, R2
        """
        col_len = tensor.size(1)
        tensor = tensor.unsqueeze(dim=1).repeat(1, num_reps, 1)
        tensor = tensor.view(-1, col_len)
        return tensor

        #get heading to pooling
    def get_heading_difference(self, obs_traj_rel, _start, _end, dim):
        start = _start
        end = _end
        
        heading_mask = nn.init.eye_(torch.empty(end-start, end-start))
        delta_x = obs_traj_rel[0,start:end, 0]  - obs_traj_rel[-1,start:end, 0] 
        delta_y = obs_traj_rel[0,start:end, 1]  - obs_traj_rel[-1,start:end, 1] 
        theta = torch.atan2(delta_x, delta_y)
        for t in range(0,end-start-1):
            for p in range(t+1, end-start):
                angle = abs(torch.atan2(torch.sin(theta[t]-theta[p]), torch.cos(theta[t]-theta[p])))
                heading_mask[t,p] = heading_mask[p,t] =torch.cos(angle)
        
        mask = heading_mask.unsqueeze(2).repeat(1,1,dim).cuda()
        
        
        return mask

    def forward(self, h_states, seq_start_end, end_pos, obs_traj_rel=None, _obs_delta_poolnet=None):
        """
        Inputs:
        - h_states: Tensor of shape (num_layers, batch, h_dim)
        - seq_start_end: A list of tuples which delimit sequences within batch
        - end_pos: Tensor of shape (batch, 2)
        Output:
        - pool_h: Tensor of shape (batch, bottleneck_dim)
        """
        pool_h = []
        for _, (start, end) in enumerate(seq_start_end):
            start = start.item()
            end = end.item()
            num_ped = end - start
            curr_hidden = h_states.view(-1, self.h_dim)[start:end]
            curr_end_pos = end_pos[start:end]
            # Repeat -> H1, H2, H1, H2
            curr_hidden_1 = curr_hidden.repeat(num_ped, 1)
            # Repeat position -> P1, P2, P1, P2
            curr_end_pos_1 = curr_end_pos.repeat(num_ped, 1)
            # Repeat position -> P1, P1, P2, P2
            curr_end_pos_2 = self.repeat(curr_end_pos, num_ped)
            curr_rel_pos = curr_end_pos_1 - curr_end_pos_2
            curr_rel_embedding = self.spatial_embedding(curr_rel_pos)
            mlp_h_input = torch.cat([curr_rel_embedding, curr_hidden_1], dim=1)
            curr_pool_h = self.mlp_pre_pool(mlp_h_input)

            #heading_mask
            if self.group_pooling is True:
                
                #mask = self.get_heading_difference(obs_traj_rel, start, end, self.bottleneck_dim)
                assert _obs_delta_poolnet is not None
                tracked_heading = _obs_delta_poolnet[3,start:end,0:num_ped] * torch.cos(_obs_delta_poolnet[2,start:end,0:num_ped])#torch.abs(torch.sin(_obs_delta_poolnet[2,start:end,0:num_ped]))
                #* torch.cos(_obs_delta_poolnet[2,start:end,0:num_ped])
                mask = tracked_heading.unsqueeze(2).repeat(1,1,self.bottleneck_dim)
                
               
            else:
                mask = torch.ones(num_ped, num_ped,self.bottleneck_dim).cuda()

            curr_pool_h = curr_pool_h.view(num_ped, num_ped, -1).mul(mask).max(1)[0]
            pool_h.append(curr_pool_h)
        pool_h = torch.cat(pool_h, dim=0)
        return pool_h


class SocialPooling(nn.Module):
    """Current state of the art pooling mechanism:
    http://cvgl.stanford.edu/papers/CVPR16_Social_LSTM.pdf"""
    def __init__(
        self, h_dim=64, activation='relu', batch_norm=True, dropout=0.0,
        neighborhood_size=2.0, grid_size=8, pool_dim=None
    ):
        super(SocialPooling, self).__init__()
        self.h_dim = h_dim
        self.grid_size = grid_size
        self.neighborhood_size = neighborhood_size
        if pool_dim:
            mlp_pool_dims = [grid_size * grid_size * h_dim, pool_dim]
        else:
            mlp_pool_dims = [grid_size * grid_size * h_dim, h_dim]

        self.mlp_pool = make_mlp(
            mlp_pool_dims,
            activation=activation,
            batch_norm=batch_norm,
            dropout=dropout
        )

    def get_bounds(self, ped_pos):
        top_left_x = ped_pos[:, 0] - self.neighborhood_size / 2
        top_left_y = ped_pos[:, 1] + self.neighborhood_size / 2
        bottom_right_x = ped_pos[:, 0] + self.neighborhood_size / 2
        bottom_right_y = ped_pos[:, 1] - self.neighborhood_size / 2
        top_left = torch.stack([top_left_x, top_left_y], dim=1)
        bottom_right = torch.stack([bottom_right_x, bottom_right_y], dim=1)
        return top_left, bottom_right

    def get_grid_locations(self, top_left, other_pos):
        cell_x = torch.floor(
            ((other_pos[:, 0] - top_left[:, 0]) / self.neighborhood_size) *
            self.grid_size)
        cell_y = torch.floor(
            ((top_left[:, 1] - other_pos[:, 1]) / self.neighborhood_size) *
            self.grid_size)
        grid_pos = cell_x + cell_y * self.grid_size
        return grid_pos

    def repeat(self, tensor, num_reps):
        """
        Inputs:
        -tensor: 2D tensor of any shape
        -num_reps: Number of times to repeat each row
        Outpus:
        -repeat_tensor: Repeat each row such that: R1, R1, R2, R2
        """
        col_len = tensor.size(1)
        tensor = tensor.unsqueeze(dim=1).repeat(1, num_reps, 1)
        tensor = tensor.view(-1, col_len)
        return tensor

    def forward(self, h_states, seq_start_end, end_pos):
        """
        Inputs:
        - h_states: Tesnsor of shape (num_layers, batch, h_dim)
        - seq_start_end: A list of tuples which delimit sequences within batch.
        - end_pos: Absolute end position of obs_traj (batch, 2)
        Output:
        - pool_h: Tensor of shape (batch, h_dim)
        """
        pool_h = []
        for _, (start, end) in enumerate(seq_start_end):
            start = start.item()
            end = end.item()
            num_ped = end - start
            grid_size = self.grid_size * self.grid_size
            curr_hidden = h_states.view(-1, self.h_dim)[start:end]
            curr_hidden_repeat = curr_hidden.repeat(num_ped, 1)
            curr_end_pos = end_pos[start:end]
            curr_pool_h_size = (num_ped * grid_size) + 1
            curr_pool_h = curr_hidden.new_zeros((curr_pool_h_size, self.h_dim))
            # curr_end_pos = curr_end_pos.data
            top_left, bottom_right = self.get_bounds(curr_end_pos)

            # Repeat position -> P1, P2, P1, P2
            curr_end_pos = curr_end_pos.repeat(num_ped, 1)
            # Repeat bounds -> B1, B1, B2, B2
            top_left = self.repeat(top_left, num_ped)
            bottom_right = self.repeat(bottom_right, num_ped)

            grid_pos = self.get_grid_locations(
                    top_left, curr_end_pos).type_as(seq_start_end)
            # Make all positions to exclude as non-zero
            # Find which peds to exclude
            x_bound = ((curr_end_pos[:, 0] >= bottom_right[:, 0]) +
                       (curr_end_pos[:, 0] <= top_left[:, 0]))
            y_bound = ((curr_end_pos[:, 1] >= top_left[:, 1]) +
                       (curr_end_pos[:, 1] <= bottom_right[:, 1]))

            within_bound = x_bound + y_bound
            within_bound[0::num_ped + 1] = 1  # Don't include the ped itself
            within_bound = within_bound.view(-1)

            # This is a tricky way to get scatter add to work. Helps me avoid a
            # for loop. Offset everything by 1. Use the initial 0 position to
            # dump all uncessary adds.
            grid_pos += 1
            total_grid_size = self.grid_size * self.grid_size
            offset = torch.arange(
                0, total_grid_size * num_ped, total_grid_size
            ).type_as(seq_start_end)

            offset = self.repeat(offset.view(-1, 1), num_ped).view(-1)
            grid_pos += offset
            grid_pos[within_bound != 0] = 0
            grid_pos = grid_pos.view(-1, 1).expand_as(curr_hidden_repeat)

            curr_pool_h = curr_pool_h.scatter_add(0, grid_pos,
                                                  curr_hidden_repeat)
            curr_pool_h = curr_pool_h[1:]
            pool_h.append(curr_pool_h.view(num_ped, -1))

        pool_h = torch.cat(pool_h, dim=0)
        pool_h = self.mlp_pool(pool_h)
        return pool_h


class TrajectoryDiscriminator(nn.Module):
    def __init__(
        self, obs_len, pred_len, embedding_dim=64, h_dim=64, mlp_dim=1024,
        num_layers=1, activation='relu', batch_norm=True, dropout=0.0,
        d_type='local'
    ):
        super(TrajectoryDiscriminator, self).__init__()

        self.obs_len = obs_len
        self.pred_len = pred_len
        self.seq_len = obs_len + pred_len
        self.mlp_dim = mlp_dim
        self.h_dim = h_dim
        self.d_type = d_type

        self.encoder = Encoder(
            embedding_dim=embedding_dim,
            h_dim=h_dim,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            dropout=dropout
        )

        real_classifier_dims = [h_dim, mlp_dim, 1]
        self.real_classifier = make_mlp(
            real_classifier_dims,
            activation=activation,
            batch_norm=batch_norm,
            dropout=dropout
        )
        if d_type == 'global':
            mlp_pool_dims = [h_dim + embedding_dim, mlp_dim, h_dim]
            self.pool_net = PoolHiddenNet(
                embedding_dim=embedding_dim,
                h_dim=h_dim,
                mlp_dim=mlp_pool_dims,
                bottleneck_dim=h_dim,
                activation=activation,
                batch_norm=batch_norm
            )

    def forward(self, traj, traj_rel, seq_start_end=None):
        """
        Inputs:
        - traj: Tensor of shape (obs_len + pred_len, batch, 2)
        - traj_rel: Tensor of shape (obs_len + pred_len, batch, 2)
        - seq_start_end: A list of tuples which delimit sequences within batch
        Output:
        - scores: Tensor of shape (batch,) with real/fake scores
        """
        final_h = self.encoder(traj_rel)
        # Note: In case of 'global' option we are using start_pos as opposed to
        # end_pos. The intution being that hidden state has the whole
        # trajectory and relative postion at the start when combined with
        # trajectory information should help in discriminative behavior.
        if self.d_type == 'local':
            classifier_input = final_h.squeeze()
        else:
            classifier_input = self.pool_net(
                final_h.squeeze(), seq_start_end, traj[0]
            )
        scores = self.real_classifier(classifier_input)
        return scores


class TrajectoryIntention(nn.Module):
    def __init__(
        self, obs_len, pred_len, embedding_dim=64, encoder_h_dim=64, 
        decoder_h_dim=128, mlp_dim=1024, num_layers=1, activation='relu', 
        batch_norm=True, dropout=0.0, bottleneck_dim=1024, goal_dim=(2, )
    ):
        super(TrajectoryIntention, self).__init__()

        self.obs_len = obs_len
        self.pred_len = pred_len
        self.mlp_dim = mlp_dim
        self.num_layers = num_layers
        self.encoder_h_dim = encoder_h_dim
        self.decoder_h_dim = decoder_h_dim
        self.embedding_dim = embedding_dim
        self.bottleneck_dim = 1024
        self.goal_dim = goal_dim

        self.encoder = Encoder(
            embedding_dim=embedding_dim,
            h_dim=encoder_h_dim,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            dropout=dropout
        )

        self.decoder = Decoder(
            pred_len,
            embedding_dim=embedding_dim,
            h_dim=decoder_h_dim,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            pool_every_timestep=False,
            dropout=dropout,
            bottleneck_dim=bottleneck_dim,
            activation=activation,
        )

        if self.goal_dim[0] == 0:
            self.goal_dim = None
        else:
            self.goal_first_dim = goal_dim[0]

        if self.goal_dim or self.encoder_h_dim != self.decoder_h_dim:
            # We need mlp to make dim consistent
            mlp_decoder_context_dims = [
                encoder_h_dim, mlp_dim, decoder_h_dim - self.goal_first_dim
            ]

            self.mlp_decoder_context = make_mlp(
                mlp_decoder_context_dims,
                activation=activation,
                batch_norm=batch_norm,
                dropout=dropout
            )
    
    def add_goal(self, _input, seq_start_end, goal_input=None):
        """
        Inputs:
        - _input: Tensor of shape (_, decoder_h_dim - noise_first_dim)
        - seq_start_end: A list of tuples which delimit sequences within batch.
        - user_noise: Generally used for inference when you want to see
        relation between different types of noise and outputs.
        Outputs:
        - decoder_h: Tensor of shape (_, decoder_h_dim)
        """
        if not self.goal_dim:
            return _input

        goal_shape = (_input.size(0), ) + self.goal_dim

        z_decoder = get_noise(goal_shape, 'inject_goal', aux_input=goal_input)

        decoder_h = torch.cat([_input, z_decoder], dim=1)

        return decoder_h

    def forward(self, obs_traj, obs_traj_rel, seq_start_end, goal_input=None, seq_len=12):
        """
        Inputs:
        - obs_traj: Tensor of shape (obs_len, 2)
        - obs_traj_rel: Tensor of shape (obs_len, batch, 2)
        - seq_start_end: A list of tuples which delimit sequences within batch.
        - aux_input: Goal information
        Output:
        - pred_traj_rel: Tensor of shape (self.pred_len, batch, 2)
        """
        batch = obs_traj_rel.size(1)
        # Encode seq
        final_encoder_h = self.encoder(obs_traj_rel)
        
        mlp_decoder_context_input = final_encoder_h.view(
                -1, self.encoder_h_dim)

        if self.goal_dim or self.encoder_h_dim != self.decoder_h_dim:
            # We need mlp to make dim consistent
            noise_input = self.mlp_decoder_context(mlp_decoder_context_input)
        else:
            noise_input = mlp_decoder_context_input
        
        decoder_h = self.add_goal(
                noise_input, seq_start_end, goal_input=goal_input)
        decoder_h = torch.unsqueeze(decoder_h, 0)

        decoder_c = torch.zeros(
            self.num_layers, batch, self.decoder_h_dim
        ).cuda()

        state_tuple = (decoder_h, decoder_c)
        last_pos = obs_traj[-1]
        last_pos_rel = obs_traj_rel[-1]
        # Predict Trajectory
        decoder_out = self.decoder(
            last_pos,
            last_pos_rel,
            state_tuple,
            seq_start_end,
            seq_len=seq_len
        )
        pred_traj_fake_rel, final_decoder_h = decoder_out

        return pred_traj_fake_rel

    def encode_obs(self, obs_traj, obs_traj_rel, seq_start_end, goal_input=None):
        """
        Inputs:
        - obs_traj: Tensor of shape (obs_len, 2)
        - obs_traj_rel: Tensor of shape (obs_len, batch, 2)
        - seq_start_end: A list of tuples which delimit sequences within batch.
        - aux_input: Goal information
        Output:
        - pred_traj_rel: Tensor of shape (self.pred_len, batch, 2)
        """
        batch = obs_traj_rel.size(1)
        # Encode seq
        final_encoder_h = self.encoder(obs_traj_rel)
        
        mlp_decoder_context_input = final_encoder_h.view(
                -1, self.encoder_h_dim)

        if self.goal_dim or self.encoder_h_dim != self.decoder_h_dim:
            # We need mlp to make dim consistent
            noise_input = self.mlp_decoder_context(mlp_decoder_context_input)
        else:
            noise_input = mlp_decoder_context_input
        
        decoder_h = self.add_goal(
                noise_input, seq_start_end, goal_input=goal_input)
        decoder_h = torch.unsqueeze(decoder_h, 0)

        decoder_c = torch.zeros(
            self.num_layers, batch, self.decoder_h_dim
        ).cuda()

        state_tuple = (decoder_h, decoder_c)
        last_pos = obs_traj[-1]
        last_pos_rel = obs_traj_rel[-1]

        return last_pos, last_pos_rel, state_tuple

    def step_decode(self, last_pos, last_pos_rel, state_tuple, seq_start_end):
        """
        Inputs:
        - last_pos: Tensor of shape (batch, 2)
        - last_pos_rel: Tensor of shape (batch, 2)
        - state_tuple: (hh, ch) each tensor of shape (num_layers, batch, h_dim)
        - seq_start_end: A list of tuples which delimit sequences within batch
        Output:
        - pred_pos: tensor of shape (1, batch, 2)
        """
        rel_pos, state_tuple = self.decoder.step_forward(
            #last_pos,
            #last_pos_rel,
            torch.zeros(last_pos.size()).cuda(),   # Start with zero social force
            torch.zeros(last_pos_rel.size()).cuda(),
            state_tuple,
            seq_start_end,
        )
        return rel_pos, state_tuple

    
# use intention output as social input
class LateAttentionFullGenerator(nn.Module):
    def __init__(
        self, obs_len, pred_len, embedding_dim=64, encoder_h_dim=64,
        decoder_h_dim=128, mlp_dim=1024, num_layers=1, noise_dim=(0, ),
        noise_type='gaussian', noise_mix_type='ped', pooling_type=None,
        group_pooling=False, pool_every_timestep=True, dropout=0.0, bottleneck_dim=1024,
        activation='relu', batch_norm=True, neighborhood_size=2.0, grid_size=8, goal_dim=(2,), spatial_dim=True
    ):
        super(LateAttentionFullGenerator, self).__init__()

        if pooling_type and pooling_type.lower() == 'none':
            pooling_type = None

        self.spatial_dim=spatial_dim
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.mlp_dim = mlp_dim
        self.encoder_h_dim = encoder_h_dim
        self.decoder_h_dim = decoder_h_dim
        self.embedding_dim = embedding_dim
        self.noise_dim = noise_dim
        self.num_layers = num_layers
        self.noise_type = noise_type
        self.noise_mix_type = noise_mix_type
        self.pooling_type = pooling_type
        self.noise_first_dim = 0
        self.pool_every_timestep = pool_every_timestep
        self.bottleneck_dim = 1024
        self.goal_dim = goal_dim
        self.group_pooling = group_pooling
        
        self.intention_encoder = Encoder(
            embedding_dim=embedding_dim,
            h_dim=encoder_h_dim,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            dropout=dropout
        )

        self.force_encoder = Encoder(
            embedding_dim=embedding_dim,
            h_dim=encoder_h_dim,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            dropout=dropout
        )

        self.intention_decoder = Decoder(
            pred_len,
            embedding_dim=embedding_dim,
            h_dim=decoder_h_dim,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            pool_every_timestep=False,
            dropout=dropout,
            bottleneck_dim=bottleneck_dim,
            activation=activation,
        )
        
        self.force_decoder = Decoder(
            pred_len,
            embedding_dim=embedding_dim,
            h_dim=decoder_h_dim,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            pool_every_timestep=pool_every_timestep,
            dropout=dropout,
            bottleneck_dim=bottleneck_dim,
            activation=activation,
            batch_norm=batch_norm,
            pooling_type=pooling_type,
            grid_size=grid_size,
            neighborhood_size=neighborhood_size,
            spatial_dim=2 if spatial_dim else decoder_h_dim
        )

        if pooling_type == 'pool_net':
            self.pool_net = PoolHiddenNet(
                embedding_dim=self.embedding_dim,
                h_dim=encoder_h_dim,
                mlp_dim=mlp_dim,
                bottleneck_dim=bottleneck_dim,
                activation=activation,
                batch_norm=batch_norm,
                group_pooling=group_pooling
            )
        elif pooling_type == 'spool':
            self.pool_net = SocialPooling(
                h_dim=encoder_h_dim,
                activation=activation,
                batch_norm=batch_norm,
                dropout=dropout,
                neighborhood_size=neighborhood_size,
                grid_size=grid_size
            )

        
        if self.noise_dim[0] == 0:
            self.noise_dim = None
        else:
            self.noise_first_dim = noise_dim[0]

        # Decoder Hidden
        if pooling_type:
            input_dim = encoder_h_dim + bottleneck_dim
        else:
            input_dim = encoder_h_dim
        
        if self.goal_dim[0] == 0:
            self.goal_dim = None
        else:
            self.goal_first_dim = goal_dim[0]

        if self.force_mlp_decoder_needed():
            mlp_decoder_context_dims = [
                input_dim, mlp_dim, decoder_h_dim - self.noise_first_dim
            ]


            self.force_mlp_decoder_context = make_mlp(
                mlp_decoder_context_dims,
                activation=activation,
                batch_norm=batch_norm,
                dropout=dropout
            )

        if self.intention_mlp_decoder_needed():
            mlp_decoder_context_dims = [
                encoder_h_dim, mlp_dim, decoder_h_dim - self.goal_first_dim
            ]

            self.intention_mlp_decoder_context = make_mlp(
                mlp_decoder_context_dims,
                activation=activation,
                batch_norm=batch_norm,
                dropout=dropout
            )

        self.attention_mlp = nn.Linear(2*decoder_h_dim, 2)
        # nn.init.kaiming_normal_(self.attention_mlp.weight)

    def add_noise(self, _input, seq_start_end, user_noise=None, aux_input=None):
        """
        Inputs:
        - _input: Tensor of shape (_, decoder_h_dim - noise_first_dim)
        - seq_start_end: A list of tuples which delimit sequences within batch.
        - user_noise: Generally used for inference when you want to see
        relation between different types of noise and outputs.
        Outputs:
        - decoder_h: Tensor of shape (_, decoder_h_dim)
        """
        if not self.noise_dim:
            return _input

        if self.noise_mix_type == 'global':
            # Randomize one noise per "batch" (multiple trajectories)
            noise_shape = (seq_start_end.size(0), ) + self.noise_dim
        else:
            # Randomize one noise per "traj"
            noise_shape = (_input.size(0), ) + self.noise_dim

        if user_noise is not None:
            z_decoder = user_noise
        else:
            z_decoder = get_noise(noise_shape, self.noise_type, aux_input=aux_input)

        if self.noise_mix_type == 'global':
            _list = []
            for idx, (start, end) in enumerate(seq_start_end):
                start = start.item()
                end = end.item()
                _vec = z_decoder[idx].view(1, -1)
                _to_cat = _vec.repeat(end - start, 1)
                _list.append(torch.cat([_input[start:end], _to_cat], dim=1))
            decoder_h = torch.cat(_list, dim=0)
            return decoder_h

        decoder_h = torch.cat([_input, z_decoder], dim=1)

        return decoder_h

    def add_goal(self, _input, seq_start_end, goal_input=None):
        """
        Inputs:
        - _input: Tensor of shape (_, decoder_h_dim - noise_first_dim)
        - seq_start_end: A list of tuples which delimit sequences within batch.
        - user_noise: Generally used for inference when you want to see
        relation between different types of noise and outputs.
        Outputs:
        - decoder_h: Tensor of shape (_, decoder_h_dim)
        """
        if not self.goal_dim:
            return _input

        goal_shape = (_input.size(0), ) + self.goal_dim

        z_decoder = get_noise(goal_shape, 'inject_goal', aux_input=goal_input)

        decoder_h = torch.cat([_input, z_decoder], dim=1)

        return decoder_h

    def force_mlp_decoder_needed(self):
        if (
            self.noise_dim or self.pooling_type or
            self.encoder_h_dim != self.decoder_h_dim
        ):
            return True
        else:
            return False
    
    def intention_mlp_decoder_needed(self):
        if (
            self.goal_dim or self.encoder_h_dim != self.decoder_h_dim
        ):
            return True
        else:
            return False
    
    
    def forward(self, obs_traj, obs_traj_rel, seq_start_end, _obs_delta_in=None, aux_input=None, user_noise=None, goal_input=None, seq_len=8, gt_rel=None):
        """
        Inputs:
        - obs_traj: Tensor of shape (obs_len, batch, 2)
        - obs_traj_rel: Tensor of shape (obs_len, batch, 2)
        - seq_start_end: A list of tuples which delimit sequences within batch.
        - user_noise: Generally used for inference when you want to see
        relation between different types of noise and outputs.
        Output:
        - pred_traj_rel: Tensor of shape (self.pred_len, batch, 2)
        """
        
        batch_size = obs_traj_rel.size(1)
        # Encode seq
        force_final_encoder_h = self.force_encoder(obs_traj_rel)
        intention_final_encoder_h = self.intention_encoder(obs_traj_rel)

        # Pool States
        if self.pooling_type:
            end_pos = obs_traj[-1, :, :]
            if self.group_pooling is True:
                pool_h = self.pool_net(force_final_encoder_h, seq_start_end, end_pos, obs_traj_rel, _obs_delta_in)
            else:
                pool_h = self.pool_net(force_final_encoder_h, seq_start_end, end_pos, obs_traj_rel)
           
            # Construct input hidden states for decoder
            force_mlp_decoder_context_input = torch.cat(
                [force_final_encoder_h.view(-1, self.encoder_h_dim), pool_h], dim=1)
        else:
            force_mlp_decoder_context_input = force_final_encoder_h.view(
                -1, self.encoder_h_dim)

        intention_mlp_decoder_context_input = intention_final_encoder_h.view(-1, self.encoder_h_dim)

        # Add Noise
        if self.force_mlp_decoder_needed():
            noise_input = self.force_mlp_decoder_context(force_mlp_decoder_context_input)
        else:
            noise_input = force_mlp_decoder_context_input
        force_decoder_h = self.add_noise(
            noise_input, seq_start_end, aux_input=aux_input, user_noise=user_noise)
        force_decoder_h = torch.unsqueeze(force_decoder_h, 0)

        force_decoder_c = torch.zeros(
            self.num_layers, batch_size, self.decoder_h_dim
        ).cuda()

        if self.intention_mlp_decoder_needed():
            noise_input = self.intention_mlp_decoder_context(intention_mlp_decoder_context_input)
        else:
            noise_input = intention_mlp_decoder_context_input
        
        intention_decoder_h = self.add_goal(
                noise_input, seq_start_end, goal_input=goal_input)
        intention_decoder_h = torch.unsqueeze(intention_decoder_h, 0)

        intention_decoder_c = torch.zeros(
            self.num_layers, batch_size, self.decoder_h_dim
        ).cuda()

            

        force_state_tuple = (force_decoder_h, force_decoder_c)
        intention_state_tuple = (intention_decoder_h, intention_decoder_c)

        last_pos = obs_traj[-1]
        last_pos_rel = obs_traj_rel[-1]
        # Predict Trajectory

        ret = []
        attention = []
        intent = []
        social = []
        
        for t in range(seq_len):

            intention_rel_pos, intention_state_tuple = self.intention_decoder.step_forward(last_pos, last_pos_rel, intention_state_tuple, seq_start_end)

            intention_pos = intention_rel_pos + obs_traj[0]
            
            if self.spatial_dim:
                force_rel_pos, force_state_tuple = self.force_decoder.step_forward(intention_pos, intention_rel_pos, force_state_tuple, seq_start_end)
            else:
                force_rel_pos, force_state_tuple = self.force_decoder.step_forward(intention_pos, intention_state_tuple[0], force_state_tuple, seq_start_end)

            attention_score = self.attention_mlp(torch.cat([force_state_tuple[0].view(-1, self.decoder_h_dim), intention_state_tuple[0].view(-1, self.decoder_h_dim)], dim=1))

            attention_score = torch.nn.functional.softmax(attention_score, dim=1)

            last_pos_rel = force_rel_pos*attention_score[:, 0].view(-1,1) + intention_rel_pos*attention_score[:,1].view(-1,1)
            #ret.append(last_pos_rel)
            #print (attention_score[0,:])
            ret.append(last_pos_rel)
            attention.append(attention_score)
            intent.append(intention_rel_pos)
            social.append(force_rel_pos)

            if gt_rel is not None:
                last_pos_rel = gt_rel[t, :, :]

            last_pos = last_pos_rel+obs_traj[0]


        
        return (torch.stack(ret), [torch.stack(attention), torch.stack(intent), torch.stack(social)])
    
