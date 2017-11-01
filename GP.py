import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

class GP:
    def __init__(self, kernel, kernel_params, noise_var):
        self.kernel = kernel
        self.params = kernel_params
        self.noise_var = noise_var
        self.x = None
        self.y = None
        self.y_mean = 0
        self.K = None
        self.inv_K = None

    def get_covariance(self, x, *params):
        return self.kernel(x, x, *params) + self.noise_var * np.identity(x.shape[0])

    def fit(self, x, y):
        self.K = self.get_covariance(x, *self.params)
        self.inv_K = np.linalg.inv(self.K)
        self.x = x
        self.y_mean = np.mean(y)
        self.y = y - self.y_mean

    def nll_params_with_gradient(self, params):
        # Note - negative log likelihood, with gradient returned
        # Bishop 6.69
        K = self.get_covariance(self.x, *params)
        inv_K = np.linalg.inv(K)
        
        sign, abslogdet = np.linalg.slogdet(K)
        logdet = np.power(abslogdet, sign)
        objective = 0.5 * (logdet + self.y.T.dot(inv_K).dot(self.y
            ) - len(self.x) * np.log(2*np.pi))

        # Bishop 6.70
        gradients = np.array([-0.5 * (-np.trace(inv_K.dot(param_gradient)) 
            + self.y.T.dot(inv_K).dot(param_gradient).dot(inv_K).dot(self.y)).squeeze()
            for param_gradient in self.kernel.gradients(self.x, self.x, *params)])

        return objective.squeeze(), np.array(gradients)

    def set_ml_parameters(self):
        res = minimize(self.nll_params_with_gradient, x0=self.params, jac=True )
        if not res.success:
            print(f'Warning: {res.message}')
        print(f"Params changed from {self.params} -> {res.x}")
        self.params = res.x
        self.K = self.get_covariance(self.x, *self.params)
        self.inv_K = np.linalg.inv(self.K)

    def posterior(self, new_x):
        if self.x is None:
            # We are sampling from prior
            mu = np.zeros(len(new_x))
            cov = self.get_covariance(new_x, *self.params) 
        else:
            k_new = self.kernel(self.x, new_x, *self.params)
            mu = k_new.T.dot(self.inv_K).dot(self.y).flatten()
            cov = self.get_covariance(new_x, *self.params) - k_new.T.dot(self.inv_K).dot(k_new) 
       
        return mu + self.y_mean, cov

class SQEKernel:
    def __call__(self, x1, x2, *params):
        sigma, ls = params
        return (sigma**2) * np.exp((-0.5 * np.power(cdist(x1, x2), 2)) * ls**2)

    def gradients(self, x1, x2, *params):
        sigma, ls = params
        gradients = []
        kernel = self.__call__(x1, x2, *params)
        # sigma
        gradients.append(2 * kernel / sigma)
        # ls
        gradients.append(kernel * (-np.power(cdist(x1, x2), 2) * ls))
        return gradients
