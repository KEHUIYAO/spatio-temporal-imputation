import numpy as np
import torch
import torch.nn as nn

from torch.nn.parameter import Parameter
from typing import Tuple, Union, Optional
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter


def cal_mae(
    predictions: Union[np.ndarray, torch.Tensor, list],
    targets: Union[np.ndarray, torch.Tensor, list],
    masks: Optional[Union[np.ndarray, torch.Tensor, list]] = None,
) -> Union[float, torch.Tensor]:
    """Calculate the Mean Absolute Error between ``predictions`` and ``targets``.
    ``masks`` can be used for filtering. For values==0 in ``masks``,
    values at their corresponding positions in ``predictions`` will be ignored.

    Parameters
    ----------
    predictions :
        The prediction data to be evaluated.

    targets :
        The target data for helping evaluate the predictions.

    masks :
        The masks for filtering the specific values in inputs and target from evaluation.
        When given, only values at corresponding positions where values ==1 in ``masks`` will be used for evaluation.

    Examples
    --------

    >>> import numpy as np
    >>> from pypots.utils.metrics import cal_mae
    >>> targets = np.array([1, 2, 3, 4, 5])
    >>> predictions = np.array([1, 2, 1, 4, 6])
    >>> mae = cal_mae(predictions, targets)

    mae = 0.6 here, the error is from the 3rd and 5th elements and is :math:`|3-1|+|5-6|=3`, so the result is 3/5=0.6.

    If we want to prevent some values from MAE calculation, e.g. the first three elements here,
    we can use ``masks`` to filter out them:

    >>> masks = np.array([0, 0, 0, 1, 1])
    >>> mae = cal_mae(predictions, targets, masks)

    mae = 0.5 here, the first three elements are ignored, the error is from the 5th element and is :math:`|5-6|=1`,
    so the result is 1/2=0.5.

    """
    assert type(predictions) == type(targets), (
        f"types of inputs and target must match, but got"
        f"type(inputs)={type(predictions)}, type(target)={type(targets)}"
    )
    lib = np if isinstance(predictions, np.ndarray) else torch
    if masks is not None:
        return lib.sum(lib.abs(predictions - targets) * masks) / (
            lib.sum(masks) + 1e-12
        )
    else:
        return lib.mean(lib.abs(predictions - targets))

def torch_parse_delta(missing_mask: torch.Tensor) -> torch.Tensor:
    """Generate time-gap (delta) matrix from missing masks.
    Please refer to :cite:`che2018GRUD` for its math definition.

    Parameters
    ----------
    missing_mask :
        Binary masks indicate missing values. Shape of [n_steps, n_features] or [n_samples, n_steps, n_features]

    Returns
    -------
    delta
        Delta matrix indicates time gaps of missing values.
    """

    def cal_delta_for_single_sample(mask: torch.Tensor) -> torch.Tensor:
        """calculate single sample's delta. The sample's shape is [n_steps, n_features]."""
        d = []
        for step in range(n_steps):
            if step == 0:
                d.append(torch.zeros(1, n_features, device=device))
            else:
                d.append(
                    torch.ones(1, n_features, device=device) + (1 - mask[step]) * d[-1]
                )
        d = torch.concat(d, dim=0)
        return d

    device = missing_mask.device
    if len(missing_mask.shape) == 2:
        n_steps, n_features = missing_mask.shape
        delta = cal_delta_for_single_sample(missing_mask)
    else:
        n_samples, n_steps, n_features = missing_mask.shape
        delta_collector = []
        for m_mask in missing_mask:
            delta = cal_delta_for_single_sample(m_mask)
            delta_collector.append(delta.unsqueeze(0))
        delta = torch.concat(delta_collector, dim=0)

    return delta


class FeatureRegression(nn.Module):
    """The module used to capture the correlation between features for imputation.

    Attributes
    ----------
    W : tensor
        The weights (parameters) of the module.

    b : tensor
        The bias of the module.

    m (buffer) : tensor
        The mask matrix, a squire matrix with diagonal entries all zeroes while left parts all ones.
        It is applied to the weight matrix to mask out the estimation contributions from features themselves.
        It is used to help enhance the imputation performance of the network.

    Parameters
    ----------
    input_size : the feature dimension of the input
    """

    def __init__(self, input_size: int):
        super().__init__()
        self.W = Parameter(torch.Tensor(input_size, input_size))
        self.b = Parameter(torch.Tensor(input_size))

        m = torch.ones(input_size, input_size) - torch.eye(input_size, input_size)
        self.register_buffer("m", m)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        std_dev = 1.0 / math.sqrt(self.W.size(0))
        self.W.data.uniform_(-std_dev, std_dev)
        if self.b is not None:
            self.b.data.uniform_(-std_dev, std_dev)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward processing of the NN module.

        Parameters
        ----------
        x : tensor,
            the input for processing

        Returns
        -------
        output: tensor,
            the processed result containing imputation from feature regression

        """
        output = F.linear(x, self.W * Variable(self.m), self.b)
        return output

class TemporalDecay(nn.Module):
    """The module used to generate the temporal decay factor gamma in the original paper.

    Attributes
    ----------
    W: tensor,
        The weights (parameters) of the module.
    b: tensor,
        The bias of the module.

    Parameters
    ----------
    input_size : int,
        the feature dimension of the input

    output_size : int,
        the feature dimension of the output

    diag : bool,
        whether to product the weight with an identity matrix before forward processing
    """

    def __init__(self, input_size: int, output_size: int, diag: bool = False):
        super().__init__()
        self.diag = diag
        self.W = Parameter(torch.Tensor(output_size, input_size))
        self.b = Parameter(torch.Tensor(output_size))

        if self.diag:
            assert input_size == output_size
            m = torch.eye(input_size, input_size)
            self.register_buffer("m", m)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        std_dev = 1.0 / math.sqrt(self.W.size(0))
        self.W.data.uniform_(-std_dev, std_dev)
        if self.b is not None:
            self.b.data.uniform_(-std_dev, std_dev)

    def forward(self, delta: torch.Tensor) -> torch.Tensor:
        """Forward processing of the NN module.

        Parameters
        ----------
        delta : tensor, shape [batch size, sequence length, feature number]
            The time gaps.

        Returns
        -------
        gamma : array-like, same shape with parameter `delta`, values in (0,1]
            The temporal decay factor.
        """
        if self.diag:
            gamma = F.relu(F.linear(delta, self.W * Variable(self.m), self.b))
        else:
            gamma = F.relu(F.linear(delta, self.W, self.b))
        gamma = torch.exp(-gamma)
        return gamma

class RITS(nn.Module):
    """model RITS: Recurrent Imputation for Time Series

    Attributes
    ----------
    n_steps :
        sequence length (number of time steps)

    n_features :
        number of features (input dimensions)

    rnn_hidden_size :
        the hidden size of the RNN cell

    device :
        specify running the model on which device, CPU/GPU

    rnn_cell :
        the LSTM cell to model temporal data

    temp_decay_h :
        the temporal decay module to decay RNN hidden state

    temp_decay_x :
        the temporal decay module to decay data in the raw feature space

    hist_reg :
        the temporal-regression module to project RNN hidden state into the raw feature space

    feat_reg :
        the feature-regression module

    combining_weight :
        the module used to generate the weight to combine history regression and feature regression

    Parameters
    ----------
    n_steps :
        sequence length (number of time steps)

    n_features :
        number of features (input dimensions)

    rnn_hidden_size :
        the hidden size of the RNN cell

    device :
        specify running the model on which device, CPU/GPU

    """

    def __init__(
        self,
        n_steps: int,
        n_features: int,
        rnn_hidden_size: int,
        device: Union[str, torch.device],
    ):
        super().__init__()
        self.n_steps = n_steps
        self.n_features = n_features
        self.rnn_hidden_size = rnn_hidden_size
        self.device = device

        self.rnn_cell = nn.LSTMCell(self.n_features * 2, self.rnn_hidden_size)
        self.temp_decay_h = TemporalDecay(
            input_size=self.n_features, output_size=self.rnn_hidden_size, diag=False
        )
        self.temp_decay_x = TemporalDecay(
            input_size=self.n_features, output_size=self.n_features, diag=True
        )
        self.hist_reg = nn.Linear(self.rnn_hidden_size, self.n_features)
        self.feat_reg = FeatureRegression(self.n_features)
        self.combining_weight = nn.Linear(self.n_features * 2, self.n_features)

    def impute(
        self, inputs: dict, direction: str
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The imputation function.
        Parameters
        ----------
        inputs :
            Input data, a dictionary includes feature values, missing masks, and time-gap values.

        direction :
            A keyword to extract data from parameter `data`.

        Returns
        -------
        imputed_data :
            [batch size, sequence length, feature number]

        hidden_states: tensor,
            [batch size, RNN hidden size]

        reconstruction_loss :
            reconstruction loss

        """
        values = inputs[direction]["X"]  # feature values
        masks = inputs[direction]["missing_mask"]  # missing masks
        deltas = inputs[direction]["deltas"]  # time-gap values

        # create hidden states and cell states for the lstm cell
        hidden_states = torch.zeros(
            (values.size()[0], self.rnn_hidden_size), device=values.device
        )
        cell_states = torch.zeros(
            (values.size()[0], self.rnn_hidden_size), device=values.device
        )

        estimations = []
        reconstruction_loss = torch.tensor(0.0).to(values.device)

        # imputation period
        for t in range(self.n_steps):
            # data shape: [batch, time, features]
            x = values[:, t, :]  # values
            m = masks[:, t, :]  # mask
            d = deltas[:, t, :]  # delta, time gap

            gamma_h = self.temp_decay_h(d)
            gamma_x = self.temp_decay_x(d)

            hidden_states = hidden_states * gamma_h  # decay hidden states
            x_h = self.hist_reg(hidden_states)
            # reconstruction_loss += cal_mae(x_h, x, m)

            x_c = m * x + (1 - m) * x_h

            z_h = self.feat_reg(x_c)
            # reconstruction_loss += cal_mae(z_h, x, m)

            alpha = torch.sigmoid(self.combining_weight(torch.cat([gamma_x, m], dim=1)))

            c_h = alpha * z_h + (1 - alpha) * x_h
            # reconstruction_loss += cal_mae(c_h, x, m)

            c_c = m * x + (1 - m) * c_h
            estimations.append(c_h.unsqueeze(dim=1))

            inputs = torch.cat([c_c, m], dim=1)
            hidden_states, cell_states = self.rnn_cell(
                inputs, (hidden_states, cell_states)
            )

        estimations = torch.cat(estimations, dim=1)
        imputed_data = masks * values + (1 - masks) * estimations
        return imputed_data, hidden_states, reconstruction_loss

    def forward(self, inputs: dict, direction: str = "forward") -> dict:
        """Forward processing of the NN module.
        Parameters
        ----------
        inputs :
            The input data.

        direction :
            A keyword to extract data from parameter `data`.

        Returns
        -------
        dict,
            A dictionary includes all results.

        """
        imputed_data, hidden_state, reconstruction_loss = self.impute(inputs, direction)
        # for each iteration, reconstruction_loss increases its value for 3 times
        reconstruction_loss /= self.n_steps * 3

        ret_dict = {
            "consistency_loss": torch.tensor(
                0.0, device=imputed_data.device
            ),  # single direction, has no consistency loss
            "reconstruction_loss": reconstruction_loss,
            "imputed_data": imputed_data,
            "final_hidden_state": hidden_state,
        }
        return ret_dict


class BRITS(nn.Module):
    """model BRITS: Bidirectional RITS
    BRITS consists of two RITS, which take time-series data from two directions (forward/backward) respectively.

    Attributes
    ----------
    n_steps :
        sequence length (number of time steps)

    n_features :
        number of features (input dimensions)

    rnn_hidden_size :
        the hidden size of the RNN cell

    rits_f: RITS object
        the forward RITS model

    rits_b: RITS object
        the backward RITS model

    """

    def __init__(
        self,
        n_steps: int,
        n_features: int,
        rnn_hidden_size: int,
        device: Union[str, torch.device],
    ):
        super().__init__()
        # data settings
        self.n_steps = n_steps
        self.n_features = n_features
        # imputer settings
        self.rnn_hidden_size = rnn_hidden_size
        # create models
        self.rits_f = RITS(n_steps, n_features, rnn_hidden_size, device)
        self.rits_b = RITS(n_steps, n_features, rnn_hidden_size, device)

    @staticmethod
    def _get_consistency_loss(
        pred_f: torch.Tensor, pred_b: torch.Tensor
    ) -> torch.Tensor:
        """Calculate the consistency loss between the imputation from two RITS models.

        Parameters
        ----------
        pred_f :
            The imputation from the forward RITS.

        pred_b :
            The imputation from the backward RITS (already gets reverted).

        Returns
        -------
        float tensor,
            The consistency loss.

        """
        loss = torch.abs(pred_f - pred_b).mean() * 1e-1
        return loss

    @staticmethod
    def _reverse(ret: dict) -> dict:
        """Reverse the array values on the time dimension in the given dictionary.

        Parameters
        ----------
        ret :

        Returns
        -------
        dict,
            A dictionary contains values reversed on the time dimension from the given dict.

        """

        def reverse_tensor(tensor_):
            if tensor_.dim() <= 1:
                return tensor_
            indices = range(tensor_.size()[1])[::-1]
            indices = torch.tensor(
                indices, dtype=torch.long, device=tensor_.device, requires_grad=False
            )
            return tensor_.index_select(1, indices)

        for key in ret:
            ret[key] = reverse_tensor(ret[key])

        return ret

    def forward(self, batch, is_train: bool = True) -> dict:


        ###############################TODO################################
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            cut_length,
            observed_covariates
        ) = self.process_data(batch)

        if is_train == 0:
            cond_mask = gt_mask
        elif self.target_strategy != 'random':
            cond_mask = self.get_hist_mask(
                observed_mask, for_pattern_mask=for_pattern_mask
            )
        else:
            cond_mask = self.get_randmask(observed_mask)

        B, K, L = observed_data.shape


        observed_data = observed_data.permute(0, 2, 1)   # reshape observed_data to (B, L, K)
        observed_mask = observed_mask.permute(0, 2, 1)   # reshape observed_mask to (B, L, K)
        cond_mask = cond_mask.permute(0, 2, 1)  # reshape cond_mask to (B, L, K)

        target_mask = observed_mask - cond_mask

        non_missing_data = observed_data * cond_mask

        forward_missing_mask = cond_mask
        forward_X = torch.nan_to_num(non_missing_data)
        forward_delta = torch_parse_delta(forward_missing_mask)
        backward_X = torch.flip(forward_X, dims=[1])
        backward_missing_mask = torch.flip(forward_missing_mask, dims=[1])
        backward_delta = torch_parse_delta(backward_missing_mask)

        inputs = {
            "forward": {
                "X": forward_X,
                "missing_mask": forward_missing_mask,
                "delta": forward_delta,
            },
            "backward": {
                "X": backward_X,
                "missing_mask": backward_missing_mask,
                "delta": backward_delta,
            },
        }

        ###################################################################


        # Results from the forward RITS.
        ret_f = self.rits_f(inputs, "forward")
        # Results from the backward RITS.
        ret_b = self._reverse(self.rits_b(inputs, "backward"))

        imputed_data = (ret_f["imputed_data"] + ret_b["imputed_data"]) / 2

        # consistency_loss = self._get_consistency_loss(
        #     ret_f["imputed_data"], ret_b["imputed_data"]
        # )
        #
        # # `loss` is always the item for backward propagating to update the model
        # loss = (
        #     consistency_loss
        #     + ret_f["reconstruction_loss"]
        #     + ret_b["reconstruction_loss"]
        # )
        #
        # results = {
        #     "imputed_data": imputed_data,
        #     "consistency_loss": consistency_loss,
        #     "loss": loss,  # will be used for backward propagating to update the model
        # }
        #
        # return results

        residual = (observed_data - imputed_data) * target_mask
        num_eval = target_mask.sum()
        loss = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)
        return loss


    def process_data(self, batch):
        observed_data = batch["observed_data"].to(self.device).float()
        if self.covariate_size > 0 and 'observed_covariates' in batch:
            observed_covariates = batch["observed_covariates"].to(self.device).float()
        else:
            observed_covariates = None
        observed_mask = batch["observed_mask"].to(self.device).float()
        observed_tp = batch["timepoints"].to(self.device).float()
        gt_mask = batch["gt_mask"].to(self.device).float()

        cut_length = []
        if 'for_pattern_mask' in batch:
            for_pattern_mask = batch["for_pattern_mask"].to(self.device).float()
        else:
            for_pattern_mask = None

        return (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            cut_length,
            observed_covariates
        )

    def get_randmask(self, observed_mask):
        if self.missing_pattern is None:
            self.missing_pattern = 'random'

        if self.missing_pattern == 'random':  # randomly sample a mask for each batch
            rand_for_mask = torch.rand_like(observed_mask) * observed_mask
            rand_for_mask = rand_for_mask.reshape(len(rand_for_mask), -1)
            for i in range(len(observed_mask)):
                sample_ratio = np.random.rand()  # missing ratio
                num_observed = observed_mask[i].sum().item()
                num_masked = round(num_observed * sample_ratio)
                rand_for_mask[i][rand_for_mask[i].topk(num_masked).indices] = -1
            cond_mask = (rand_for_mask > 0).reshape(observed_mask.shape).float()

        elif self.missing_pattern == 'block':
            B, K, L = observed_mask.size()  # (B, K, L)
            cond_mask = observed_mask.clone()
            # each batch has a different missing ratio
            for i in range(len(observed_mask)):
                sample_ratio = np.random.rand()  # missing ratio
                expected_num_masked = round(observed_mask[i].sum().item() * sample_ratio)
                cur_num_masked = 0
                # if block size is provided in config, use it
                if 'time_block_size' in self.config['model']:
                    l_block_size = self.config['model']['time_block_size']
                else:
                    l_block_size = np.random.randint(1, L+1)

                if 'space_block_size' in self.config['model']:
                    k_block_size = self.config['model']['space_block_size']
                else:
                    k_block_size = 1

                while cur_num_masked < expected_num_masked:
                    l_start = np.random.randint(0, L-l_block_size+1)
                    k_start = np.random.randint(0, K-k_block_size+1)

                    # if cond_mask is 0, then we don't count it
                    cur_num_masked += cond_mask[i, k_start:k_start+k_block_size, l_start:l_start+l_block_size].sum().item()

                    # set the mask to 0
                    cond_mask[i, k_start:k_start + k_block_size, l_start:l_start + l_block_size] = 0


        else:
            raise NotImplementedError

        return cond_mask

    def evaluate(self, batch, nsample=1):
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            cut_length,
            observed_covariates
        ) = self.process_data(batch)

        with torch.no_grad():
            cond_mask = gt_mask
            B, K, L = observed_data.shape

            observed_data = observed_data.permute(0, 2, 1)  # reshape observed_data to (B, L, K)
            observed_mask = observed_mask.permute(0, 2, 1)  # reshape observed_mask to (B, L, K)
            cond_mask = cond_mask.permute(0, 2, 1)  # reshape cond_mask to (B, L, K)

            target_mask = observed_mask - cond_mask
            non_missing_data = observed_data * cond_mask

            forward_missing_mask = cond_mask
            forward_X = torch.nan_to_num(non_missing_data)
            forward_delta = torch_parse_delta(forward_missing_mask)
            backward_X = torch.flip(forward_X, dims=[1])
            backward_missing_mask = torch.flip(forward_missing_mask, dims=[1])
            backward_delta = torch_parse_delta(backward_missing_mask)

            inputs = {
                "forward": {
                    "X": forward_X,
                    "missing_mask": forward_missing_mask,
                    "delta": forward_delta,
                },
                "backward": {
                    "X": backward_X,
                    "missing_mask": backward_missing_mask,
                    "delta": backward_delta,
                },
            }


            # Results from the forward RITS.
            ret_f = self.rits_f(inputs, "forward")
            # Results from the backward RITS.
            ret_b = self._reverse(self.rits_b(inputs, "backward"))

            imputed_data = (ret_f["imputed_data"] + ret_b["imputed_data"]) / 2  # shape (B, L, K)


            # permute back to the original order BxKxL
            predicted = imputed_data.permute(0, 2, 1)
            observed_data = observed_data.permute(0, 2, 1)
            target_mask = target_mask.permute(0, 2, 1)
            observed_mask = observed_mask.permute(0, 2, 1)


            # add one more dimension to make sure the shape is (B, 1, K, L)
            predicted = predicted.unsqueeze(1)

            return predicted, observed_data, target_mask, observed_mask, observed_tp
