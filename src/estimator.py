import numpy as np
import pandas as pd
from globals import *
from utils import *
from tqdm import tqdm


class Estimator:
    '''
    Estimator assumes a stochastic variable x which has some self similarity and also some covariance with another signal. 
    The distribution of x is assumed dependent on past values of x
    The distribution of x is also assumed dependent on values for y

    The estimator trains on datasets of x and y, analyzing covariance and autocorrelations.
    
    '''

    exact: bool
    n_xlags: int

    sample_data: np.ndarray
    estimated_data: np.ndarray
    
    nx = 0
    ny = 0
    
    n_data: int

    covariances: np.ndarray
    means: np.array

    def __init__(self, dependent_data: np.ndarray, independent_data: np.ndarray = None, 
                 x_labels = [], y_labels = [], n_xlags = 0, n_ylags=0, is_exact = False):

        dependent_data, independent_data, x_labels, y_labels = self.sanitize_inputs(dependent_data, independent_data, x_labels, y_labels)
        if x_labels: assert len(x_labels) == dependent_data.shape[0]
        if y_labels: assert len(y_labels) == independent_data.shape[0]

        self.exact = is_exact
        self.n_xlags = n_xlags
        self.n_ylags = n_ylags
        self.max_lags = max(n_xlags, n_ylags)
        self.min_lags = min(n_xlags, n_ylags)
        self.sample_data = dependent_data
        self.n_data = dependent_data.shape[1]

        self.x_labels = x_labels
        self.y_labels = y_labels
        
        self.add_sample(X_sample=dependent_data, Y_sample=independent_data)
        self.calculate_estimate(Y_sample = independent_data)


    def __getitem__(self, index):
        return self.estimated_data[index]

    def __setitem__(self, index, value):
        self.estimated_data[index] = value

    def mean(self):
        return np.mean(self.estimated_data)

    def variance(self):
        return np.var(self.estimated_data)

    def __len__(self):
        return len(self.estimated_data)

    def __repr__(self):
        return f"EstimatingArray({self.estimated_data})"



    def sanitize_inputs(self, dependent_data: np.ndarray, independent_data: np.ndarray = None,  x_labels=[], y_labels=[]):

        if len(dependent_data.shape) == 1: dependent_data = dependent_data.reshape((1,-1))
        
        if independent_data is None:         independent_data = np.zeros((0, dependent_data.shape[1]))
        if len(independent_data.shape) == 1: independent_data = independent_data.reshape((1,-1))
        
        if type(x_labels) == str: x_labels = [x_labels]
        if type(y_labels) == str: y_labels = [y_labels]

        return dependent_data, independent_data, x_labels, y_labels

    def add_sample(self, X_sample: np.ndarray, Y_sample: np.ndarray):

        assert X_sample.shape[1] == Y_sample.shape[1],  f"Inconsistent lengths of dependent and independent data"
        assert X_sample.shape[1] > self.n_xlags - 1,    f"Sample data is too short for choice of lag variables"
        assert Y_sample.shape[1] == self.n_data,        f"Size inconsistency when adding input signal. Expected length {self.n_data}, received length{Y_sample.shape[1]} "
        
        self.ny = Y_sample.shape[0]
        self.nx = X_sample.shape[0]

        self.update_covariances(Y_sample)
        

    def update_covariances(self, Y_sample):

        lagged_xdata = np.zeros((self.nx*self.n_xlags, self.n_data - self.n_xlags))

        for k in range(1, self.n_xlags+1):
            lagged_xdata[self.nx*(k-1):self.nx*k, :] = self.sample_data[:,self.n_xlags-k:self.n_data-k]


        lagged_ydata = np.zeros((self.ny*self.n_ylags, self.n_data - self.n_ylags))
        for k in range(1, self.n_ylags+1):
            lagged_ydata[self.ny*(k-1):self.ny*k, :] = Y_sample[:,self.n_ylags-k:self.n_data-k]

        all_signals     = np.vstack((self.sample_data[:,self.max_lags:],
                                    lagged_xdata[:,self.max_lags - self.n_xlags:], 
                                    Y_sample[:,self.max_lags:],
                                    lagged_ydata[:,self.max_lags - self.n_ylags:])) 
        
        self.covariances = np.cov(all_signals)
        self.means = np.mean(all_signals, axis=1)
        # self.means[self.nx:self.nx+self.n_lags*self.nx] = np.repeat(self.means[:self.nx], self.n_lags)
    

    def build_stable_covariances(self, threshold=1e8):
        """
        Incrementally builds a well-conditioned covariance matrix Pyy,
        skipping lag variables that make it numerically unstable.
        
        Args:
            y_lagged: Matrix of lagged y values (shape: [n_samples, n_lags])
            x_lagged: Corresponding lagged x values (for Pxy update)
            threshold: Condition number threshold for stability
        
        Returns:
            Pyy: Well-conditioned covariance matrix
            Pxy: Corresponding cross-covariance matrix
            selected_lags: List of indices of selected lags
        """
        n = self.nx + self.n_xlags + self.ny + self.n_ylags
        selected_lags = []

        full_covariances = self.covariances[self.nx:, self.nx:]

        Pyy = np.zeros((0,0))
        Pxy = np.zeros((0, self.n_xlags + self.ny + self.n_ylags))  # Matching empty cross-matrix

        for var in range(n - self.nx):

            new_col = full_covariances[selected_lags + [var], var].reshape(-1, 1)
            new_row = new_col.T

            # Pyy_candidate = self.covariances[self.nx:self.nx+lag, self.nx:self.nx+lag]
            Pyy_candidate = np.block([
                [Pyy, new_col[:-1,:]],
                [new_row[:,:-1], new_row[-1,-1]]
            ])
            
            # Compute condition number
            cond_number = np.linalg.cond(Pyy_candidate)

            if cond_number < threshold:
                Pyy = Pyy_candidate  # Accept new column/row
                selected_lags.append(var)
        
        Pxy = self.covariances[:self.nx, np.array(selected_lags) + self.nx]

        return Pyy, Pxy, selected_lags


    def calculate_estimate(self, Y_sample: np.ndarray):
        
        if self.exact:
            self.estimated_data = self.sample_data
            self.mse = 0
            self.rmse = 0
        
        # print(self.covariances)
        Pxx = self.covariances[:self.nx, :self.nx]
        Pyy, Pxy, selected_vars = self.build_stable_covariances()
        Pyy_inv = np.linalg.inv(Pyy)

        a = self.means[:self.nx].reshape((-1, 1))
        b = self.means[selected_vars].reshape((-1, 1))

        x_est = np.zeros_like(self.sample_data)
        x_est[:,:self.max_lags] = np.repeat(self.means[:self.nx].reshape((-1,1)), self.max_lags, axis=1)
        # x_est[:,:self.n_lags] = self.sample_data[:,:self.n_lags]

        for k in range(self.max_lags, self.n_data):
            
            # if Y_sample is not None:
            y = np.vstack((np.flip(self.sample_data[:,k-self.n_xlags:k], axis=1).ravel(order='F').reshape((-1,1)), 
                           Y_sample[:,k].reshape((-1,1)),
                           np.flip(Y_sample[:,k-self.n_ylags:k], axis=1).ravel(order='F').reshape((-1,1))
                           ))[selected_vars, :]
            # else:
            #     y = x_est[:,k:k+self.n_lags].ravel(order='F').reshape((-1,1))

            conditional_expectation = a + Pxy @ Pyy_inv @ (y - b)
            
            x_est[:,k] = conditional_expectation.flatten()

        self.conditional_covariance  = Pxx - Pxy @ Pyy_inv @ Pxy.T

        # self.estimated_data = np.flip(x_est, axis=0)
        self.estimated_data = x_est

        self.mse = np.mean(np.square(self.estimated_data[:,self.max_lags:] - self.sample_data[:,self.max_lags:]))
        self.rmse = np.sqrt(self.mse)
                                      
        return
    
    def measure_performance(self, how='array'):
        
        if how=='single':
            print(f"RMSE: {rmse} \tWith expected covariance {self.conditional_covariance[i,i]}")

        elif how=='array':
            for i in range(self.nx):
                signal_name = self.x_labels[i] if self.x_labels else f"X{i}"
                rmse = np.sqrt(np.mean(np.square(self.estimated_data[i,self.max_lags:] - self.sample_data[i,self.max_lags:])))
                print(f"RMSE for {signal_name}: {rmse} \tWith expected covariance {self.conditional_covariance[i,i]}")


        return



    '''
    def estimate(self, past_vals: np.ndarray = None, Y_sample: np.ndarray = None):
        
        if len(past_vals.shape) == 1: past_vals = past_vals.reshape((self.nx, -1))
        if len(Y_sample.shape) == 1: Y_sample = Y_sample.reshape((self.ny, -1))

        Pyy, Pxy, selected_vars = self.build_stable_covariances()
        Pyy_inv = np.linalg.inv(Pyy)
        a = self.means[:self.nx].reshape((-1, 1))
        b = self.means[selected_vars].reshape((-1, 1))
        
        x_est = np.zeros((self.nx, self.n_lags + Y_sample.shape[1]))
        x_est[:,:self.n_lags] = past_vals[:,:n_lags]

        for k in range(self.n_lags, self.n_data):
            
            y = np.vstack((np.flip(x_est[:,k-self.n_lags:k].ravel(order='F')).reshape((-1,1)), Y_sample[:,k].reshape((-1,1))))[selected_vars, :]
            
            conditional = a + Pxy @ Pyy_inv @ (y - b)
            
            x_est[:,k] = conditional.flatten()

        return x_est[:,self.n_lags:]
        '''

''' #

# x = np.sin(np.linspace(0, 6*np.pi,signal_length)).reshape((1,-1))
# x = np.tile(np.array([[-1,-1,-2,-2],
                    #   [1,1,2,2]]), int(signal_length/4))


# y = x.repeat(1, axis=0)
# np.random.seed(1133)
# y = y + 0.2*np.random.randn(*y.shape)


signal_length = 100
y = np.linspace(1, 10, signal_length).reshape((1,-1))
x = np.vstack((2*y + np.random.randn(*y.shape),
                -0.2*y + 0.3*np.random.randn(*y.shape)))


fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

ax1.plot(np.arange(signal_length), x[0,:].flatten(), color='black', label='x', linewidth=3, linestyle=':')
ax2.plot(np.arange(signal_length), x[1,:].flatten(), color='black', label='x', linewidth=3, linestyle=':')

mse = {}

for lag in [0, 1, 2, 3, 4, 5, 6, 7, 8]:
    n_lags = lag
    est = Estimator(x, y, n_lags=n_lags, is_exact=False)

    # print(f"MSE: {est.mse}")
    mse[lag] = (est.mse)

    past_vals = np.array([0])

    output = est[:,:]
    # output = est.estimate(past_vals, y)

    ax1.plot(np.arange(signal_length), output[0,:].flatten(), label=f'lag: {lag}')
    ax2.plot(np.arange(signal_length), output[1,:].flatten(), label=f'lag: {lag}')
    ax1.legend()
    ax2.legend()

for lag in mse:
    print(f"MSE for lag {lag}: {mse[lag]}")

plt.show()

# '''





class EstimatorDF:

    estimated_data: np.ndarray

    def __init__(self, name, x_truth_df: pd.DataFrame, x_training_df: pd.DataFrame, y_input_df: pd.DataFrame = None,  y_training_df: pd.DataFrame = None,
                 xlags=None, ylags=None, exact=False, show_output = False):
        assert isinstance(x_truth_df, pd.DataFrame), "x_df must be a pandas DataFrame"
        assert isinstance(x_training_df, pd.DataFrame), "x_df must be a pandas DataFrame"
        if y_input_df is not None:
            assert isinstance(y_input_df, pd.DataFrame), "y_df must be a pandas DataFrame"
        else:
            y_input_df = pd.DataFrame(index=x_truth_df.index)
        if y_training_df is not None:
            assert isinstance(y_training_df, pd.DataFrame), "y_df must be a pandas DataFrame"
        else:
            y_training_df = pd.DataFrame(index=x_training_df.index)

        self.x_df = x_truth_df.copy()
        self.y_df = y_input_df.copy()
        self.x_training_df = x_training_df.copy()
        self.y_training_df = y_training_df.copy()
        self.xlags = sorted(xlags or [])
        self.ylags = sorted(ylags or [])
        self.n_xlags = len(self.xlags)
        self.n_ylags = len(self.ylags)
        self.nx = self.x_df.shape[1]
        self.ny = self.y_df.shape[1]
        self.max_lag = max(self.xlags + self.ylags + [0])
        self.max_xlag = max(self.xlags + [0])
        self.max_ylag = max(self.ylags + [0])
        self.exact = exact

        self.estimated_df = pd.DataFrame(index=x_truth_df.index, columns=x_truth_df.columns, dtype='float64')
        self.covariance_matrix = None
        self.conditional_covariance = None
        self.rmse_scores = None

        self.name = name
        self.show_output = show_output

        self.update_covariances()
        self.build_stable_covariances()
        self.calculate_estimate()

    def build_lagged_matrix(self, df, lags, prefix):
        lagged = []
        for lag in lags:
            shifted = df.shift(lag)
            shifted.columns = [f"{prefix} {col} lag {lag}" for col in df.columns]
            lagged.append(shifted)
        return pd.concat(lagged, axis=1)

    def update_covariances(self):
        lagged_x_truth = self.build_lagged_matrix(self.x_df, self.xlags, 'x')
        lagged_y_input = self.build_lagged_matrix(self.y_df, self.ylags, 'y')

        lagged_x_training = self.build_lagged_matrix(self.x_training_df, self.xlags, 'x')
        lagged_y_training = self.build_lagged_matrix(self.y_training_df, self.ylags, 'y')

        # combined = pd.concat([self.x_df, lagged_x, self.y_df, lagged_y], axis=1).dropna()
        combined_training = pd.concat([self.x_training_df, lagged_x_training, lagged_y_training], axis=1).dropna()
        combined_truth    = pd.concat([self.x_df, lagged_x_truth, lagged_y_input], axis=1).dropna()
        self.indices_truth      = combined_truth.index
        self.indices_training   = combined_training.index
        self.full_data_truth    = combined_truth
        self.full_data_training = combined_training

        self.covariance_matrix = combined_training.cov().values
        self.means = combined_training.mean().values

    def build_stable_covariances(self, threshold=1e8):
        nx = self.x_df.shape[1]
        total_vars = self.full_data_training.shape[1]

        Pyy = np.zeros((0, 0))
        Pxy = np.zeros((nx, 0))
        selected_lags = []


        with tqdm(total = total_vars - nx, desc=f"Estimator {self.name} Building stable covariance matrix") as pbar:
            for i in range(nx, total_vars):
                if Pyy.size == 0:
                    candidate = np.array([[self.covariance_matrix[i, i]]])
                else:
                    col = self.covariance_matrix[selected_lags + [i], i].reshape(-1, 1)
                    row = col.T
                    candidate = np.block([[Pyy, col[:-1]], [row[:, :-1], row[:, -1]]])

                if np.linalg.cond(candidate) < threshold:
                    selected_lags.append(i)
                    Pyy = candidate
                    new_Pxy_col = self.covariance_matrix[:nx, i].reshape(-1, 1)
                    Pxy = np.hstack([Pxy, new_Pxy_col])
                
                pbar.update(1)

        Pxx = self.covariance_matrix[:self.x_df.shape[1], :self.x_df.shape[1]]
        Pyy_inv = np.linalg.inv(Pyy)

        self.selected_vars = np.array(selected_lags)
        self.Pxx = Pxx
        self.Pxy = Pxy
        self.Pyy = Pyy
        self.Pyy_inv = Pyy_inv

        weights = Pxy @ Pyy_inv

        full_weights = np.zeros((self.nx, self.nx*self.max_xlag + self.ny * self.max_ylag))
        full_weights[:, self.selected_vars-nx] = weights

        self.weights = weights
        self.full_weights = full_weights
        self.full_weights_x = full_weights[:,:self.nx*self.max_xlag]
        self.full_weights_y = full_weights[:,self.nx*self.max_xlag:]
        return

    def calculate_estimate(self):
        if self.exact:
            self.estimated_df.loc[self.indices_truth] = self.x_df.loc[self.indices_truth]
            self.estimated_data = np.array(self.x_df.loc[self.indices_truth]).transpose()
            self.conditional_covariance = np.zeros_like(self.Pxx)
            self.conditional_variance = {col: self.conditional_covariance[i,i] for i, col in enumerate(self.x_df.columns)}
            self.rmse_scores = 0
            return

        selected_vars   = self.selected_vars
        Pxx             = self.Pxx
        Pxy             = self.Pxy
        Pyy             = self.Pyy
        Pyy_inv         = self.Pyy_inv
        weights         = self.weights

        mu_x = self.means[:self.x_df.shape[1]].reshape((-1, 1))
        mu_y = self.means[selected_vars].reshape((-1, 1))

        estimates = []
        for idx in self.indices_truth:
            y = self.full_data_truth.loc[idx].values[selected_vars].reshape((-1, 1))
            x_hat = mu_x + weights @ (y - mu_y)
            estimates.append(x_hat.flatten())

        est_array = np.array(estimates)
        self.estimated_data = est_array.T
        self.estimated_df.loc[self.indices_truth] = est_array
        self.conditional_covariance = Pxx - Pxy @ Pyy_inv @ Pxy.T
        self.conditional_variance = {col: self.conditional_covariance[i,i] for i, col in enumerate(self.x_df.columns)}

        self.rmse_scores = np.sqrt(np.mean((self.estimated_df.loc[self.indices_truth] - self.x_df.loc[self.indices_truth]) ** 2))

    def calculate_future_prediction(self, true_x, past_x, y):
        '''
        Calculate prediction of future values

        Inputs:
            - past_x:   Initial values of x.
            - input_y:  values of y
            - past_y:   Initial values of y.

        Outputs: 
            - x_pred:   pd.DataFrame of predicted x.values for time slots for which y-values exist
        
        '''

        # if self.exact:
        #     self.estimated_df.loc[self.indices_truth] = self.x_df.loc[self.indices_truth]
        #     return
                
        selected_vars  = self.selected_vars

        past_x_array    = np.array(past_x).transpose()
        y_array         = np.array(y).transpose()

        n_pastx = past_x_array.shape[1]

        # assert n_pastx >= self.max_xlag, f"past_x is too short. Expected at least length {self.max_xlag}, but got a length of {past_x_array.shape[1]}"
        # assert past_y.shape[1] >= self.max_ylag, f"past_y is too short. Expected at least length {self.max_ylag}, but got a length of {past_y.shape[1]}"

        
        x_pred = pd.DataFrame(index=true_x.index, columns=true_x.columns, dtype='float64')

        selected_past_x = selected_vars[np.where(selected_vars <= self.max_xlag*self.nx)]
        selected_past_y = selected_vars[np.where(selected_vars >  self.max_xlag*self.nx)] - self.max_xlag

        # mu_x = self.means[:self.x_df.shape[1]].reshape((-1, 1))
        mu_x = np.array(self.x_df.mean()).reshape((1,-1))
        # mu_y = self.means[selected_vars].reshape((-1, 1))


        x_hat = np.hstack((past_x_array, np.zeros((past_x_array.shape[0], y_array.shape[1] -  n_pastx))))
        # x_hat = np.zeros_like(y_array)

        # x_mean = np.array(self.x_df.mean())
        # y_mean = np.array(self.y_df.mean())
        x_mean = np.tile(self.x_df.mean(), self.max_xlag)
        y_mean = np.tile(self.y_df.mean(), self.max_ylag)

        full_weights_x = np.flip(self.full_weights_x, axis=1)
        full_weights_y = np.flip(self.full_weights_y, axis=1)
        # full_weights_x = np.array(self.full_weights_x)
        # full_weights_y = np.array(self.full_weights_y)

        full_weights_x = np.zeros_like(self.full_weights_x)
        for i in range(self.nx): full_weights_x[:,i::self.nx] = np.flip(self.full_weights_x[:,i::self.nx], axis=1)
        full_weights_y = np.zeros_like(self.full_weights_y)
        for i in range(self.ny): full_weights_y[:,i::self.ny] = np.flip(self.full_weights_y[:,i::self.ny], axis=1)


        # estimates = []
        for k in range(n_pastx, x_hat.shape[1]):
            # y = self.full_data_truth.loc[idx].values[selected_vars].reshape((-1, 1))

            x_vals = x_hat[:,   k-self.max_xlag:k].transpose().ravel()
            y_vals = y_array[:, k-self.max_ylag:k].transpose().ravel()

            # y = np.hstack((x_hat[:,k - selected_past_x], y_array[:, k - selected_past_y])).reshape((-1,1))

            # x_hat = mu_x + Pxy @ Pyy_inv @ (y - mu_y)
            next_x = mu_x.flatten() + full_weights_x @ (x_vals - x_mean) + full_weights_y @ (y_vals - y_mean)
            
            x_hat[:,k] = next_x
            
            # estimates.append(x_hat.flatten())

        # est_array = np.array(x_hat)
        # self.estimated_data = est_array.T
        x_pred.loc[true_x.index] = x_hat[:, n_pastx:].transpose()
        # self.conditional_covariance = Pxx - Pxy @ Pyy_inv @ Pxy.T
        # self.conditional_variance = {col: self.conditional_covariance[i,i] for i, col in enumerate(self.x_df.columns)}

        # self.rmse_scores = np.sqrt(np.mean((self.estimated_df.loc[self.indices_truth] - self.x_df.loc[self.indices_truth]) ** 2))

        print(f'{self.name} RMSE: {rmse(true_x, x_pred, axis=0)}')

        return x_pred
        


    def measure_performance(self):
        for col in self.x_df.columns:
            rmse = np.sqrt(np.mean(np.square(self.estimated_df[col] - self.x_df[col])))
            print(f"Estimator {self.name} RMSE for {col}: {rmse:.4f}")

    def show_estimator_profile(self, n_vars = 10):
        print(f"\n{self.name} estimator profile:")
        
        selected_vars = self.selected_vars
        
        for timeseries in self.conditional_variance:
            print(f"{timeseries} \tEstimator standard deviation: {np.sqrt(self.conditional_variance[timeseries])}")

        for i, col in enumerate(self.x_df.columns):
            
            weights = self.weights

            sorted_weights_idx = np.argsort(np.abs(weights[i,:]))[-n_vars:]

            print(f"Most important inputs for {col}:")
            for idx in reversed(sorted_weights_idx):
                var = selected_vars[idx]
                print(f"\tInput: {self.full_data_training.columns[var]}\tWeight {weights[i,idx]:.4f}")
                # print(f"\tInput: {self.full_data.columns[var]}\tCovariance {self.covariance_matrix[i,var]:.4f}")

