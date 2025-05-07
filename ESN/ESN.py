# Distribution
from   typing import Any, Callable, Tuple

# Community
import numpy as np
from   scipy import linalg
import matplotlib.pyplot as plt

#--------------------------------------------------------------------------
class InputLayer:
#--------------------------------------------------------------------------
    def __init__(self, in_features: int, n_nodes: int,
                 scale: float = 1.0, bias: float = 1.0) -> None:
        self.in_features = in_features
        self.n_nodes     = n_nodes
        self.scale       = scale
        self.W           = None
        self.b           = bias

        self.init_weights()

    def init_weights(self) -> None:
        # uniform distribution [-1, 1]
        self.W = np.random.uniform(-1, 1,
                                   size=(self.in_features,
                                         self.n_nodes)) * self.scale

    def reset(self) -> None:
        self.init_weights()

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.dot(x, self.W) + self.b

#--------------------------------------------------------------------------
class OutputLayer:
#--------------------------------------------------------------------------
    def __init__(self, n_nodes: int, out_features: int,
                 bias: float = 1.0) -> None:
        self.n_nodes      = n_nodes
        self.out_features = out_features
        self.W            = None
        self.b            = bias

        self.init_weights()

    def init_weights(self) -> None:
        # uniform distribution [-1, 1]
        self.W = np.random.uniform(-1, 1,
                                   size=(self.n_nodes, self.out_features))

    def reset(self) -> None:
        self.init_weights()

    def update_weights(self, weights: np.ndarray) -> None:
        # update weights of output layer
        assert weights.shape == (self.n_nodes, self.out_features)
        self.W = weights

    def __call__(self, r_t: np.ndarray) -> np.ndarray:
        return np.dot(r_t, self.W) + self.b

#--------------------------------------------------------------------------
class ReservoirLayer:
#--------------------------------------------------------------------------
    def __init__(self, num_nodes: int, leak_rate: float,
                 spectral_radius: float = 0.0, activation=np.tanh) -> None:
        self.num_nodes = num_nodes

        # leakage rate [0, 1]
        self.leak_rate       = leak_rate
        self.spectral_radius = spectral_radius
        self.activation      = activation
        self.W               = None

        self.init_weights()

    def init_weights(self) -> None:
        weights = np.random.normal(0, 1,
            self.num_nodes * self.num_nodes).reshape([self.num_nodes,
                                                      self.num_nodes])
        # Initial spectral_radius
        W_lambda = max( abs(linalg.eigvals(weights)) )

        # Scale initial spectral radius by specified value
        self.spectral_radius = W_lambda / self.spectral_radius

        # Apply spectral radius scaling (eigenvalues are linear op)
        self.W = weights / self.spectral_radius

        print( 'ESN:ReservoirLayer.init_weights : W spectral radius: ' +\
               f' {max( abs(linalg.eigvals(self.W)) ):.6}' )

    def reset(self) -> None:
        self.init_weights()

    def __call__(self, r_t: np.ndarray, u_t: np.ndarray) -> np.ndarray:
        return (1 - self.leak_rate) * r_t + self.leak_rate * \
            self.activation(np.dot(r_t, self.W) + u_t)

#--------------------------------------------------------------------------
class ESN:
#--------------------------------------------------------------------------
    """Echo State Network using Reservoir Computing"""

    def __init__(
        self,
        num_inputs:      int,
        num_outputs:     int,
        num_resv_nodes:  int,
        leak_rate:       float = 0.5,
        spectral_radius: float = 0.0,
        activation:      Callable = np.tanh,
        input_bias:      float = 1.0,
        output_bias:     float = 0.0,
        seed:            int = 0
    ) -> None:
        """

        Parameters:
            num_inputs (int): number of unites for the input layer
            num_outputs (int): number of unites for the output layer
            num_resv_nodes (int): number of reservoir nodes
            leak_rate (float): the leaky rate for reservoir layer
            spectral_radius (float): spectral radius for reservoir layer
            activation (callable): activation function for reservoir nodes
            input_bias (float): bias for input layer (default 1.0)

        Note : spectral radius is deﬁned as the largest absolute eigenvalue
               of the reservoir weight matrix W.

        Code adapted from: 
            ReservoirComputing: Implementing Reservoir Computing Networks
            for Predicting Dynamic Systems}, Michael Hu,
            https://github.com/michaelnny/ReservoirComputing,
            version = 1.0.0, 2023
        """

        assert 0 < num_inputs
        assert 0 < num_outputs
        assert 0 < num_resv_nodes
        assert 0 < leak_rate <= 1
        assert 0 <= spectral_radius
        assert 0 <= input_bias <= 1
        assert 0 <= output_bias <= 1

        self.num_inputs      = num_inputs
        self.num_resv_nodes  = num_resv_nodes
        self.num_outputs     = num_outputs
        self.leak_rate       = leak_rate
        self.spectral_radius = spectral_radius
        self.seed            = seed

        np.random.seed( seed )
        print( 'ESN.__init__: RNG initialized with seed ', seed )

        self.input_layer = InputLayer(num_inputs, num_resv_nodes, input_bias)

        self.resv_layer = ReservoirLayer(num_resv_nodes, leak_rate,
                                         spectral_radius, activation)

        self.output_layer = OutputLayer(num_resv_nodes, num_outputs, output_bias)

        self.reservoir_states = []

    #---------------------------------------------------------------------
    def reset(self):
        self.input_layer.reset()
        self.resv_layer.reset()
        self.output_layer.reset()

        self.reservoir_states = []

    #---------------------------------------------------------------------
    def train(
        self,
        train_input: np.ndarray,
        train_target: np.ndarray,
        _lambda: float = 0.1,
    ) -> None:
        """
        Train the model by run through the input data and collecting
        reservoir states, then update the weights of the output layer.

        Parameters:
            train_input (np.ndarray): input data shape of [sequence_len, N],
                where N is the number of features we feed to the input layer.
            train_target (np.ndarray): target data shape of [sequence_len, M],
                where M is the number of features we want the model to predict.
            _lambda (float, optional): lambda for the Ridge Regression
            (default 0.1).
        """

        assert len(train_input) == len(train_target)
        assert len(train_input.shape) == len(train_target.shape) == 2

        T = len(train_input)
        self.reservoir_states = []

        # initialize dummy reservoir state for first timestep
        r_t = np.zeros(self.num_resv_nodes)
        # collect reservoir states
        for t in range(T):
            u_t = self.input_layer(train_input[t])
            r_tp1 = self.resv_layer(r_t, u_t)
            self.reservoir_states.append(r_tp1)
            r_t = r_tp1

        # update output layer weights
        reservoir_states = np.vstack(self.reservoir_states)
        self.update_output_weights(reservoir_states, train_target, _lambda)

    #---------------------------------------------------------------------
    def update_output_weights(self, reservoir_states: np.ndarray,
                              target: np.ndarray, _lambda: float) -> None:
        """Compute the output weights analytically"""
        # Ridge Regression
        E_lambda = np.identity(self.num_resv_nodes) * _lambda
        inv_x = np.linalg.inv(np.dot(reservoir_states.T,
                                     reservoir_states) + E_lambda)
        # update weights of output layer
        out_weights = np.dot(np.dot(inv_x, reservoir_states.T), target)
        # make sure have the dimension
        out_weights = out_weights.reshape(self.output_layer.W.shape)

        # TODO: what about bias of the output layer????
        self.output_layer.update_weights(out_weights)

    #---------------------------------------------------------------------
    def predict(self, input_data: np.ndarray,
                true_target: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Use the trained model to perform prediction

        Parameters:
            input_data (numpy.ndarray): input data shape of [sequence_len, N],
                where N is the number of features we feed to the input layer
            true_target (numpy.ndarray): target data shape of [sequence_len, M],
                where M is the number of features we want the model to predict

        Returns:
            tuple:
                pred_target (numpy.ndarray): a 2D numpy.ndarray with
                shape of [sequence_len, M] contains the predicted values
                mse (numpy.ndarray): a 1D numpy.ndarray with shape of [M]
                contains the MSE for each predicted dimension
        """
        assert len(input_data) == len(true_target)
        assert len(input_data.shape) == len(true_target.shape) == 2

        T = len(input_data)
        pred_target = []

        # initialize dummy reservoir state for first timestep
        r_t = np.zeros(self.num_resv_nodes)
        for t in range(T):
            u_t = self.input_layer(input_data[t])
            r_tp1 = self.resv_layer(r_t, u_t)
            x_tp1 = self.output_layer(r_tp1)

            pred_target.append(x_tp1)
            r_t = r_tp1

        pred_target = np.vstack(pred_target)
        assert pred_target.shape == true_target.shape

        # compute MSE
        # skip first step as we're using dummy reservoir state
        squared_diff = (pred_target[1:] - true_target[1:]) ** 2
        mse = np.mean(squared_diff, axis=0)

        return pred_target, mse

    #---------------------------------------------------------------------
    def predict_autonomous(self, input_data: np.ndarray,
        true_target: np.ndarray, burnin: int = 1) -> Tuple[np.ndarray, float]:
        """
        Use the trained model to perform autonomous prediction.

        Args:
            input_data (numpy.ndarray): input data shape of [sequence_len, N],
                where N is the number of features we feed to the input layer.
            true_target (numpy.ndarray): target data shape of [sequence_len, N],
                where N is the number of features we want the model to predict.
            burnin (int, optional): number of timesteps we use data from
            input_data, instead of use self-predicted as input to the
            model (default 1).

        Returns:
            tuple:
                pred_target (numpy.ndarray): a 2D numpy.ndarray with shape
                of [sequence_len, M] contains the predicted values.
                mse (numpy.ndarray): a 1D numpy.ndarray with shape of [M]
                contains the MSE for each predicted dimension.
        """

        assert len(input_data.shape) == len(true_target.shape) == 2

        T = len(true_target)
        pred_target = []

        # initialize dummy reservoir state for first timestep
        r_t = np.zeros(self.num_resv_nodes)

        #x_t = input_data[0]        
        x_t = input_data[0,:] # JP ?

        for t in range(T):
            u_t   = self.input_layer(x_t)
            r_tp1 = self.resv_layer(r_t, u_t)
            x_tp1 = self.output_layer(r_tp1)

            pred_target.append(x_tp1)

            r_t = r_tp1
            if t >= burnin:
                x_t = x_tp1 # JP This requires N == M to feedback inputs
            else:
                x_t = input_data[t,:]

        pred_target = np.vstack(pred_target)

        # compute MSE
        # skip first step as we're using dummy reservoir state
        squared_diff = (pred_target[1:] - true_target[1:]) ** 2
        mse = np.mean(squared_diff, axis=0)

        return pred_target, mse
