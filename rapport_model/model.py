#######################################################################################################################
# Project: Deep Virtual Rapport Agent (rapport model)
#
#     Jan Ondras (jo951030@gmail.com)
#     Institute for Creative Technologies, University of Southern California
#     April-October 2019
#
#######################################################################################################################
# PyTorch rapport models.
#     Both unimodal and multimodal variants.
#     Currently implemented:
#         GRU-based base classifier,
#         Temporal Convolutional Network (TCN) classifier
#######################################################################################################################


import torch
from TCN.TCN.tcn import TemporalConvNet

device = torch.device('cuda')


#######################################################################################################################
# Unimodal models
#######################################################################################################################


class UnimodalBaseClassifier(torch.nn.Module):
    """
    GRU-based base classifier for one modality.
    """

    def __init__(self, opt):
        """
        Args:
            opt: commandline options
        """

        super(UnimodalBaseClassifier, self).__init__()

        if opt.modality == 'vision':
            self.feature_dim = opt.vision_feature_dim
            self.rnn_layer_dim = opt.vision_rnn_layer_dim
        elif opt.modality == 'speech':
            self.feature_dim = opt.speech_feature_dim
            self.rnn_layer_dim = opt.speech_rnn_layer_dim
        else:
            print(f'Model is not implemented for modality {opt.modality}')

        self.rnn_layer_num = opt.rnn_layer_num
        self.rnn_dropout_rate = opt.rnn_dropout_rate
        self.bidirectional = opt.bidirectional

        self.fc_layer_num = opt.fc_layer_num
        self.fc_dropout_rate = opt.fc_dropout_rate
        self.output_dim = opt.class_num

        self.batch_size = opt.batch_size

        # RNN layers
        self.rnn = torch.nn.GRU(input_size=self.feature_dim,
                                hidden_size=self.rnn_layer_dim,
                                num_layers=self.rnn_layer_num,
                                bidirectional=self.bidirectional,
                                dropout=self.rnn_dropout_rate,
                                batch_first=True)

        # Fully-connected layers: each layer has half the dimensionality of the previous layer
        assert self.rnn_layer_dim >= 2 ** self.fc_layer_num

        self.fc_layers = torch.nn.ModuleList()
        fc_input_dim = self.rnn_layer_dim

        for i in range(self.fc_layer_num - 1):
            fc_output_dim = int(fc_input_dim / 2)
            self.fc_layers.append(torch.nn.Sequential(
                torch.nn.Linear(fc_input_dim, fc_output_dim),
                torch.nn.ReLU(),
                torch.nn.Dropout(self.fc_dropout_rate)
            ))
            fc_input_dim = fc_output_dim

        # Last layer: sigmoid activation and no dropout
        self.last_fc_layer = torch.nn.Linear(fc_input_dim, self.output_dim)
        self.sigmoid = torch.nn.Sigmoid()

        # Model parameters
        self.optim_params = []
        self.optim_params += list(self.rnn.parameters())
        for i in range(len(self.fc_layers)):
            self.optim_params += list(self.fc_layers[i].parameters())
        self.optim_params += list(self.last_fc_layer.parameters())

    def init_hidden(self):
        """Initialize RNN hidden state (done manually for online prediction, since we want to preserve the RNN hidden
        state between batches)"""

        self.h_n = torch.zeros(self.rnn_layer_num, self.batch_size, self.rnn_layer_dim).cuda()

    def forward(self, seq):
        """Forward pass through the model.

        Args:
            seq (3D array): Input sequences with shape (batch_size, sequence_length, n_features)

        Returns:
            (tuple): array of output logits and array of corresponding predictions (both of shape (batch_size,
            n_targets))
        """

        # Initial hidden state for each element in the batch is zero => don't keep state between batches (stateless)
        x, _ = self.rnn(seq)
        # Take the last timestep only
        x = x[:, -1, :]
        for i in range(len(self.fc_layers)):
            x = self.fc_layers[i](x)
        y_logits = self.last_fc_layer(x)
        y_classes = self.sigmoid(y_logits).round()

        return y_logits, y_classes

    def predict_online(self, seq):
        """Forward pass through the model during online prediction.

        Assumes the batch size of 1 and input sequence_length of 1.
        RNN state is preserved between batches.

        Args:
            seq (3D array): Input sequences with shape (1, 1, n_features)

        Returns:
            (tuple): array of output logits and array of corresponding predictions (both of shape (1, n_targets))
        """

        assert self.batch_size == 1, f'Online prediction requires the batch size ({self.batch_size}) to be one!'
        assert seq.shape[0] == 1, f'Online prediction requires the batch size ({self.batch_size}) to be one!'
        assert seq.shape[1] == 1, f'Online prediction inputs one timestep at a time!'

        # For online prediction, preserve state between batches (assuming the batch_size is 1)
        x, self.h_n = self.rnn(seq, self.h_n)
        # Take the last timestep only
        x = x[:, -1, :]
        for i in range(len(self.fc_layers)):
            x = self.fc_layers[i](x)
        y_logits = self.last_fc_layer(x)
        y_classes = self.sigmoid(y_logits).round()

        return y_logits, y_classes


class UnimodalTCNClassifier(torch.nn.Module):
    """
    Classifier based on Temporal Convolutional Network (TCN) for one modality.
    """

    def __init__(self, opt):
        """
        Args:
            opt: commandline options
        """
        super(UnimodalTCNClassifier, self).__init__()

        if opt.modality == 'vision':
            self.feature_dim = opt.vision_feature_dim
            self.rnn_layer_dim = opt.vision_rnn_layer_dim
        elif opt.modality == 'speech':
            self.feature_dim = opt.speech_feature_dim
            self.rnn_layer_dim = opt.speech_rnn_layer_dim
        else:
            print(f'Model is not implemented for modality {opt.modality}')

        self.rnn_layer_num = opt.rnn_layer_num
        self.rnn_dropout_rate = opt.rnn_dropout_rate

        self.tcn_num_inputs = self.feature_dim
        # Numbers of hidden units for each layer/level
        self.tcn_num_channels = [self.rnn_layer_dim] * self.rnn_layer_num
        self.tcn_kernel_size = opt.tcn_kernel_size

        self.fc_layer_num = opt.fc_layer_num
        self.fc_dropout_rate = opt.fc_dropout_rate
        self.output_dim = opt.class_num

        self.batch_size = opt.batch_size

        # TCN layers
        self.tcn = TemporalConvNet(num_inputs=self.tcn_num_inputs, num_channels=self.tcn_num_channels,
                                   kernel_size=self.tcn_kernel_size, dropout=self.rnn_dropout_rate)

        # Fully-connected layers: each layer has half the dimensionality of the previous layer
        assert self.rnn_layer_dim >= 2 ** self.fc_layer_num

        self.fc_layers = torch.nn.ModuleList()
        fc_input_dim = self.rnn_layer_dim

        for i in range(self.fc_layer_num - 1):
            fc_output_dim = int(fc_input_dim / 2)
            self.fc_layers.append(torch.nn.Sequential(
                torch.nn.Linear(fc_input_dim, fc_output_dim),
                torch.nn.ReLU(),
                torch.nn.Dropout(self.fc_dropout_rate)
            ))
            fc_input_dim = fc_output_dim

        # Last layer: sigmoid activation and no dropout
        self.last_fc_layer = torch.nn.Linear(fc_input_dim, self.output_dim)
        self.sigmoid = torch.nn.Sigmoid()

        # Model parameters
        self.optim_params = []
        self.optim_params += list(self.tcn.parameters())
        for i in range(len(self.fc_layers)):
            self.optim_params += list(self.fc_layers[i].parameters())
        self.optim_params += list(self.last_fc_layer.parameters())

    def forward(self, seq):
        """Forward pass through the model.

        Args:
            seq (3D array): Input sequences with shape (batch_size, sequence_length, n_features)

        Returns:
            (tuple): array of output logits and array of corresponding predictions (both of shape (batch_size,
            n_targets))
        """

        # TCN layer inputs data with shape (batch_size, n_input_channels, sequence_length)
        # TCN layer returns data with shape (batch_size, n_output_channels, sequence_length)
        # => Swap input channels (features) dimension with sequence length dimension
        seq = seq.permute(0, 2, 1)
        x = self.tcn(seq)
        # Take the last timestep only
        x = x[:, :, -1]
        for i in range(len(self.fc_layers)):
            x = self.fc_layers[i](x)
        y_logits = self.last_fc_layer(x)
        y_classes = self.sigmoid(y_logits).round()

        return y_logits, y_classes

    def predict_online(self, seq):
        """Forward pass through the model during online prediction.

        With TCN, need to accumulate past sequence_length frames (can't do one timestep at a time).
        Assumes the batch size of 1.

        Args:
            seq (3D array): Input sequences with shape (1, sequence_length, n_features)

        Returns:
            (tuple): array of output logits and array of corresponding predictions (both of shape (1, n_targets))
        """

        assert self.batch_size == 1, f'Online prediction requires the batch size ({self.batch_size}) to be one!'
        assert seq.shape[0] == 1, f'Online prediction requires the batch size ({self.batch_size}) to be one!'

        return self.forward(seq)


#######################################################################################################################
# Multimodal models
#######################################################################################################################


class UnimodalSubmodel(torch.nn.Module):
    """
    GRU-based submodel for one modality.

    Used by MultimodalBaseClassifier.
    """

    def __init__(self, modality, opt):
        """
        Args:
            modality (str): Modality: vision or speech
            opt: commandline options
        """
        super(UnimodalSubmodel, self).__init__()

        if modality == 'vision':
            self.feature_dim = opt.vision_feature_dim
            self.rnn_layer_dim = opt.vision_rnn_layer_dim
        elif modality == 'speech':
            self.feature_dim = opt.speech_feature_dim
            self.rnn_layer_dim = opt.speech_rnn_layer_dim
        else:
            print(f'Sub-model is not implemented for {modality}')

        self.rnn_layer_num = opt.rnn_layer_num
        self.rnn_dropout_rate = opt.rnn_dropout_rate
        self.bidirectional = opt.bidirectional

        self.batch_size = opt.batch_size

        # Layer parameters
        self.rnn = torch.nn.GRU(input_size=self.feature_dim,
                                hidden_size=self.rnn_layer_dim,
                                num_layers=self.rnn_layer_num,
                                bidirectional=self.bidirectional,
                                dropout=self.rnn_dropout_rate,
                                batch_first=True)

        # Model parameters
        self.optim_params = []
        self.optim_params += list(self.rnn.parameters())

    def init_hidden(self):
        """Initialize RNN hidden state (done manually for online prediction, since we want to preserve the RNN hidden
        state between batches)"""

        self.h_n = torch.zeros(self.rnn_layer_num, self.batch_size, self.rnn_layer_dim).cuda()

    def forward(self, seq):
        """Forward pass through the model.

        Args:
            seq (3D array): Input sequences with shape (batch_size, sequence_length, n_features)

        Returns:
            (3D array): Output from GRU, with shape (batch_size, sequence_length, self.rnn_layer_dim)
        """

        # Initial hidden state for each element in the batch is zero => don't keep state between batches (stateless)
        y, _ = self.rnn(seq)

        return y

    def predict_online(self, seq):
        """Forward pass through the model during online prediction.

        Assumes the batch size of 1 and input sequence_length of 1.
        RNN state is preserved between batches.

        Args:
            seq (3D array): Input sequences with shape (1, 1, n_features)

        Returns:
            (3D array): Output from GRU, with shape (1, 1, self.rnn_layer_dim)
        """

        # For online prediction, preserve state between batches (assuming the batch_size is 1)
        assert self.batch_size == 1, f'Online prediction requires the batch size ({self.batch_size}) to be one!'
        assert seq.shape[0] == 1, f'Online prediction requires the batch size ({self.batch_size}) to be one!'
        assert seq.shape[1] == 1, f'Online prediction inputs one timestep at a time!'

        y, self.h_n = self.rnn(seq, self.h_n)

        return y


class MultimodalBaseClassifier(torch.nn.Module):
    """
    GRU-based base classifier fusing multiple modalities.
    """

    def __init__(self, opt):
        """
        Args:
            opt: commandline options
        """

        super(MultimodalBaseClassifier, self).__init__()

        self.embedding_dim = opt.vision_rnn_layer_dim + opt.speech_rnn_layer_dim
        self.output_dim = opt.class_num

        self.fc_layer_num = opt.fc_layer_num
        self.fc_dropout_rate = opt.fc_dropout_rate
        self.output_dim = opt.class_num

        # RNN submodules
        self.vision_subnet = UnimodalSubmodel('vision', opt)
        self.speech_subnet = UnimodalSubmodel('speech', opt)

        # Fully-connected layers: each layer has half the dimensionality of the previous layer
        assert self.embedding_dim >= 2 ** self.fc_layer_num

        self.fc_layers = torch.nn.ModuleList()
        fc_input_dim = self.embedding_dim

        for i in range(self.fc_layer_num - 1):
            fc_output_dim = int(fc_input_dim / 2)
            self.fc_layers.append(torch.nn.Sequential(
                torch.nn.Linear(fc_input_dim, fc_output_dim),
                torch.nn.ReLU(),
                torch.nn.Dropout(self.fc_dropout_rate)
            ))
            fc_input_dim = fc_output_dim

        # Last layer: sigmoid activation and no dropout
        self.last_fc_layer = torch.nn.Linear(fc_input_dim, self.output_dim)
        self.sigmoid = torch.nn.Sigmoid()

        self.optim_params = []
        self.optim_params += list(self.vision_subnet.optim_params)
        self.optim_params += list(self.speech_subnet.optim_params)
        for i in range(len(self.fc_layers)):
            self.optim_params += list(self.fc_layers[i].parameters())
        self.optim_params += list(self.last_fc_layer.parameters())

    def init_hidden(self):
        """Initialize RNN hidden state (done manually for online prediction, since we want to preserve the RNN hidden
        state between batches)"""

        self.vision_subnet.init_hidden()
        self.speech_subnet.init_hidden()

    def forward(self, seq1, seq2):
        """Forward pass through the model.

        Args:
            seq1 (3D array): Vision input sequences with shape (batch_size, sequence_length, n_features)
            seq2 (3D array): Speech input sequences with shape (batch_size, sequence_length, n_features)

        Returns:
            (tuple): array of output logits and array of corresponding predictions (both of shape (batch_size,
            n_targets))
        """

        # Take the last timestep from subnetworks
        x_v = self.vision_subnet.forward(seq1)[:, -1, :]
        x_s = self.speech_subnet.forward(seq2)[:, -1, :]
        # print(x_v.shape, x_s.shape)

        x = torch.cat((x_v, x_s), 1)
        for i in range(len(self.fc_layers)):
            x = self.fc_layers[i](x)
        y_logits = self.last_fc_layer(x)
        y_classes = self.sigmoid(y_logits).round()

        return y_logits, y_classes

    def predict_online(self, seq1, seq2):
        """Forward pass through the model during online prediction.

        Assumes the batch size of 1 and input sequence_length of 1.
        RNN state is preserved between batches.

        Args:
            seq1 (3D array): Vision input sequences with shape (1, 1, n_features)
            seq2 (3D array): Speech input sequences with shape (1, 1, n_features)

        Returns:
            (tuple): array of output logits and array of corresponding predictions (both of shape (1, n_targets))
        """

        # Take the last timestep from subnetworks
        x_v = self.vision_subnet.predict_online(seq1)[:, -1, :]
        x_s = self.speech_subnet.predict_online(seq2)[:, -1, :]
        # print(x_v.shape, x_s.shape)

        x = torch.cat((x_v, x_s), 1)
        for i in range(len(self.fc_layers)):
            x = self.fc_layers[i](x)
        y_logits = self.last_fc_layer(x)
        y_classes = self.sigmoid(y_logits).round()

        return y_logits, y_classes


class UnimodalTCNSubmodel(torch.nn.Module):
    """
    Temporal Convolutional Network (TCN)-based submodel for one modality.

    Used by MultimodalTCNClassifier.
    """

    def __init__(self, modality, opt):
        """
        Args:
            modality (str): Modality: vision or speech
            opt: commandline options
        """

        super(UnimodalTCNSubmodel, self).__init__()

        if modality == 'vision':
            self.feature_dim = opt.vision_feature_dim
            self.rnn_layer_dim = opt.vision_rnn_layer_dim
        elif modality == 'speech':
            self.feature_dim = opt.speech_feature_dim
            self.rnn_layer_dim = opt.speech_rnn_layer_dim
        else:
            print(f'Sub-model is not implemented for {modality}')

        self.rnn_layer_num = opt.rnn_layer_num
        self.rnn_dropout_rate = opt.rnn_dropout_rate

        self.tcn_num_inputs = self.feature_dim
        # Numbers of hidden units for each layer/level
        self.tcn_num_channels = [self.rnn_layer_dim] * self.rnn_layer_num
        self.tcn_kernel_size = opt.tcn_kernel_size

        self.batch_size = opt.batch_size

        # TCN layers
        self.tcn = TemporalConvNet(num_inputs=self.tcn_num_inputs, num_channels=self.tcn_num_channels,
                                   kernel_size=self.tcn_kernel_size, dropout=self.rnn_dropout_rate)

        # Model parameters
        self.optim_params = []
        self.optim_params += list(self.tcn.parameters())

    def forward(self, seq):
        """Forward pass through the model.

        Args:
            seq (3D array): Input sequences with shape (batch_size, sequence_length, n_features)

        Returns:
            (3D array): Output from TCN, with shape (batch_size, n_output_channels, sequence_length)
        """

        # TCN layer inputs data with shape (batch_size, n_input_channels, sequence_length)
        # TCN layer returns data with shape (batch_size, n_output_channels, sequence_length)
        # => Swap input channels (features) dimension with sequence length dimension
        seq = seq.permute(0, 2, 1)
        y = self.tcn(seq)

        return y

    def predict_online(self, seq):
        """Forward pass through the model during online prediction.

        With TCN, need to accumulate past sequence_length frames (can't do one timestep at a time).
        Assumes the batch size of 1.

        Args:
            seq (3D array): Input sequences with shape (1, sequence_length, n_features)

        Returns:
            (3D array): Output from TCN, with shape (1, n_output_channels, sequence_length)
        """

        assert self.batch_size == 1, f'Online prediction requires the batch size ({self.batch_size}) to be one!'
        assert seq.shape[0] == 1, f'Online prediction requires the batch size ({self.batch_size}) to be one!'

        return self.forward(seq)


class MultimodalTCNClassifier(torch.nn.Module):
    """
    Temporal Convolutional Network (TCN)-based classifier fusing multiple modalities.
    """

    def __init__(self, opt):
        """
        Args:
            opt: commandline options
        """
        super(MultimodalTCNClassifier, self).__init__()

        self.embedding_dim = opt.vision_rnn_layer_dim + opt.speech_rnn_layer_dim
        self.output_dim = opt.class_num

        self.fc_layer_num = opt.fc_layer_num
        self.fc_dropout_rate = opt.fc_dropout_rate
        self.output_dim = opt.class_num

        # RNN submodules
        self.vision_subnet = UnimodalTCNSubmodel('vision', opt)
        self.speech_subnet = UnimodalTCNSubmodel('speech', opt)

        # Fully-connected layers: each layer has half the dimensionality of the previous layer
        assert self.embedding_dim >= 2 ** self.fc_layer_num

        self.fc_layers = torch.nn.ModuleList()
        fc_input_dim = self.embedding_dim

        for i in range(self.fc_layer_num - 1):
            fc_output_dim = int(fc_input_dim / 2)
            self.fc_layers.append(torch.nn.Sequential(
                torch.nn.Linear(fc_input_dim, fc_output_dim),
                torch.nn.ReLU(),
                torch.nn.Dropout(self.fc_dropout_rate)
            ))
            fc_input_dim = fc_output_dim

        # Last layer: sigmoid activation and no dropout
        self.last_fc_layer = torch.nn.Linear(fc_input_dim, self.output_dim)
        self.sigmoid = torch.nn.Sigmoid()

        self.optim_params = []
        self.optim_params += list(self.vision_subnet.optim_params)
        self.optim_params += list(self.speech_subnet.optim_params)
        for i in range(len(self.fc_layers)):
            self.optim_params += list(self.fc_layers[i].parameters())
        self.optim_params += list(self.last_fc_layer.parameters())

    def forward(self, seq1, seq2):
        """Forward pass through the model.

        Args:
            seq1 (3D array): Vision input sequences with shape (batch_size, sequence_length, n_features)
            seq2 (3D array): Speech input sequences with shape (batch_size, sequence_length, n_features)

        Returns:
            (tuple): array of output logits and array of corresponding predictions (both of shape (batch_size,
            n_targets))
        """

        # Take the last timestep from subnetworks
        x_v = self.vision_subnet.forward(seq1)[:, :, -1]
        x_s = self.speech_subnet.forward(seq2)[:, :, -1]
        # print(x_v.shape, x_s.shape)

        x = torch.cat((x_v, x_s), 1)
        for i in range(len(self.fc_layers)):
            x = self.fc_layers[i](x)
        y_logits = self.last_fc_layer(x)
        y_classes = self.sigmoid(y_logits).round()  # .int()

        return y_logits, y_classes

    def predict_online(self, seq1, seq2):
        """Forward pass through the model during online prediction.

        With TCN, need to accumulate past sequence_length frames (can't do one timestep at a time).
        Assumes the batch size of 1.

        Args:
            seq1 (3D array): Vision input sequences with shape (1, sequence_length, n_features)
            seq2 (3D array): Speech input sequences with shape (1, sequence_length, n_features)

        Returns:
            (tuple): array of output logits and array of corresponding predictions (both of shape (1, n_targets))
        """

        return self.forward(seq1, seq2)
