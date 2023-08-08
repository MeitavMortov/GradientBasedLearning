import numpy as np
from IMLearn import BaseModule


class L2(BaseModule):
    """
    Class representing the L2 module
    Represents the function: f(w)=||w||^2_2
    """
    def __init__(self, weights: np.ndarray = None):
        """
        Initialize a module instance
        Parameters:
        ----------
        weights: np.ndarray, default=None
            Initial value of weights
        """
        super().__init__(weights)

    def compute_output(self, **kwargs) -> np.ndarray:
        """
        Compute the output value of the L2 function at point self.weights
        Parameters
        ----------
        kwargs:
            No additional arguments are expected
        Returns
        -------
        output: ndarray of shape (1,)
            Value of function at point self.weights
        """
        # USED : https://builtin.com/data-science/vector-norms
        # ||w||^2_2 = sumi(|wi|^2)
        abs_w = np.abs(self.weights)
        squared_abs_w = abs_w**2
        return np.sum(squared_abs_w)

    def compute_jacobian(self, **kwargs) -> np.ndarray:
        """
        Compute L2 derivative with respect to self.weights at point self.weights
        Parameters
        ----------
        kwargs:
            No additional arguments are expected
        Returns
        -------
        output: ndarray of shape (n_in,)
            L2 derivative with respect to self.weights at point self.weights
        """
        # We have seen formoula in recitation 2 Example 1.3
        # ∇ f (x) = 2x.
        return 2 * self.weights


class L1(BaseModule):
    def __init__(self, weights: np.ndarray = None):
        """
        Initialize a module instance
        Parameters:
        ----------
        weights: np.ndarray, default=None
            Initial value of weights
        """
        super().__init__(weights)

    def compute_output(self, **kwargs) -> np.ndarray:
        """
        Compute the output value of the L1 function at point self.weights
        Parameters
        ----------
        kwargs:
            No additional arguments are expected
        Returns
        -------
        output: ndarray of shape (1,)
            Value of function at point self.weights
        """
        # ||w||_1 = sumi(|wi|)
        # USED : https://builtin.com/data-science/vector-norms
        abs_w = np.abs(self.weights)
        return np.sum(abs_w)

    def compute_jacobian(self, **kwargs) -> np.ndarray:
        """
        Compute L1 derivative with respect to self.weights at point self.weights
        Parameters
        ----------
        kwargs:
            No additional arguments are expected
        Returns
        -------
        output: ndarray of shape (n_in,)
            L1 derivative with respect to self.weights at point self.weights
        """
        # USED: https://math.stackexchange.com/questions/1646008/derivative-of-l-1-norm
        # ∂∥x∥1 = {s ∣∣ si = sign(xi) ∀ i∈S, ∥s∥∞≤1}
        # If wi is not differentiable  return sub gradient: [−1,1] wi = 0
        return np.sign(self.weights)

class LogisticModule(BaseModule):
    """
    Class representing the logistic regression objective function
    Represents the function: f(w) = - (1/m) sum_i^m[y*<x_i,w> - log(1+exp(<x_i,w>))]
    """
    def __init__(self, weights: np.ndarray = None):
        """
        Initialize a logistic regression module instance
        Parameters:
        ----------
        weights: np.ndarray, default=None
            Initial value of weights
        """
        super().__init__(weights)

    def compute_output(self, X: np.ndarray, y: np.ndarray, **kwargs) -> np.ndarray:
        """
        Compute the output value of the logistic regression objective function at point self.weights
        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)
            Design matrix to use when computing objective
        y: ndarray of shape(n_samples,)
            Binary labels of samples to use when computing objective
        Returns
        -------
        output: ndarray of shape (1,)
            Value of function at point self.weights
        """
        #f(w) = - (1/m) sum_i^m[y*<x_i,w> - log(1+exp(<x_i,w>))]
        M = X.shape[0]
        x_mult_w = np.matmul(X,self.weights)
        exp_x_dot_w = np.exp(x_mult_w)
        log_expression= np.log(1+exp_x_dot_w)
        sum = np.sum(y * x_mult_w - log_expression)
        return -((1/M) * sum)


    def compute_jacobian(self, X: np.ndarray, y: np.ndarray, **kwargs) -> np.ndarray:
        """
        Compute the gradient of the logistic regression objective function at point self.weights
        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)
            Design matrix to use when computing objective
        y: ndarray of shape(n_samples,)
            Binary labels of samples to use when computing objective
        Returns
        -------
        output: ndarray of shape (n_features,)
            Derivative of function with respect to self.weights at point self.weights
        """
        M = X.shape[0]
        x_mult_w = np.matmul(X, self.weights)
        exp_x_dot_w = np.exp(x_mult_w)
        divided_exp_x_dot_w = (exp_x_dot_w)/(exp_x_dot_w+1)
        y_mult_x = np.matmul(y,X)
        x_mult_divided_exp_x_dot_w = np.matmul(X.T, divided_exp_x_dot_w)
        return -(1/M) * (y_mult_x - x_mult_divided_exp_x_dot_w)


class RegularizedModule(BaseModule):
    """
    Class representing a general regularized objective function of the format:
                                    f(w) = F(w) + lambda*R(w)
    for F(w) being some fidelity function, R(w) some regularization function and lambda
    the regularization parameter
    """
    def __init__(self,
                 fidelity_module: BaseModule,
                 regularization_module: BaseModule,
                 lam: float = 1.,
                 weights: np.ndarray = None,
                 include_intercept: bool = True):
        """
        Initialize a regularized objective module instance
        Parameters:
        -----------
        fidelity_module: BaseModule
            Module to be used as a fidelity term
        regularization_module: BaseModule
            Module to be used as a regularization term
        lam: float, default=1
            Value of regularization parameter
        weights: np.ndarray, default=None
            Initial value of weights
        include_intercept: bool default=True
            Should fidelity term (and not regularization term) include an intercept or not
        """
        super().__init__()
        self.fidelity_module_, self.regularization_module_, self.lam_ = fidelity_module, regularization_module, lam
        self.include_intercept_ = include_intercept

        if weights is not None:
            self.weights = weights

    def compute_output(self, **kwargs) -> np.ndarray:
        """
        Compute the output value of the regularized objective function at point self.weights
        Parameters
        ----------
        kwargs:
            No additional arguments are expected
        Returns
        -------
        output: ndarray of shape (1,)
            Value of function at point self.weights
        """
        X = kwargs['X']
        y = kwargs['y']
        f_output = self.fidelity_module_.compute_output(X=X,y=y)
        g_output = self.regularization_module_.compute_output(X=X,y=y)
        return f_output + (self.lam_ * g_output)


    def compute_jacobian(self, **kwargs) -> np.ndarray:
        """
        Compute module derivative with respect to self.weights at point self.weights
        Parameters
        ----------
        kwargs:
            No additional arguments are expected
        Returns
        -------
        output: ndarray of shape (n_in,)
            Derivative with respect to self.weights at point self.weights
        """
        X = kwargs['X']
        y = kwargs['y']
        f_jacobian = self.fidelity_module_.compute_jacobian(X=X,y=y)
        g_jacobian = self.regularization_module_.compute_jacobian(X=X,y=y)
        if self.include_intercept_:
            return f_jacobian + (self.lam_*np.insert(g_jacobian, 0, 0))
        return f_jacobian + (self.lam_*g_jacobian)


    @property
    def weights(self):
        """
        Wrapper property to retrieve module parameter
        Returns
        -------
        weights: ndarray of shape (n_in, n_out)
        """
        return self.weights_

    @weights.setter
    def weights(self, weights: np.ndarray) -> None:
        """
        Setter function for module parameters
        In case self.include_intercept_ is set to True, weights[0] is regarded as the intercept
        and is not passed to the regularization module
        Parameters
        ----------
        weights: ndarray of shape (n_in, n_out)
            Weights to set for module
        """
        self.weights_ = weights
        self.fidelity_module_.weights = weights
        if self.include_intercept_:
            self.regularization_module_.weights = weights[1:]
        else:
            self.regularization_module_.weights = weights

