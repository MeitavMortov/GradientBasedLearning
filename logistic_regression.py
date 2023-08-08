from typing import NoReturn
import numpy as np
from IMLearn import BaseEstimator
from IMLearn.desent_methods import GradientDescent
from IMLearn.desent_methods.modules import LogisticModule, RegularizedModule, L1, L2




class LogisticRegression(BaseEstimator):
    """
    Logistic Regression Classifier
    Attributes
    ----------
    solver_: GradientDescent, default=GradientDescent()
        Descent method solver to use for the logistic regression objective optimization
    penalty_: str, default="none"
        Type of regularization term to add to logistic regression objective. Supported values
        are "none", "l1", "l2"
    lam_: float, default=1
        Regularization parameter to be used in case `self.penalty_` is not "none"
    alpha_: float, default=0.5
        Threshold value by which to convert class probability to class value
    include_intercept_: bool, default=True
        Should fitted model include an intercept or not
    coefs_: ndarray of shape (n_features,) or (n_features+1,)
        Coefficients vector fitted by linear regression. To be set in
        `LogisticRegression.fit` function.
    """

    def __init__(self,
                 include_intercept: bool = True,
                 solver: GradientDescent = GradientDescent(),
                 penalty: str = "none",
                 lam: float = 1,
                 alpha: float = .5):
        """
        Instantiate a linear regression estimator
        Parameters
        ----------
        solver: GradientDescent, default=GradientDescent()
            Descent method solver to use for the logistic regression objective optimization
        penalty: str, default="none"
            Type of regularization term to add to logistic regression objective. Supported values
            are "none", "l1", "l2"
        lam: float, default=1
            Regularization parameter to be used in case `self.penalty_` is not "none"
        alpha: float, default=0.5
            Threshold value by which to convert class probability to class value
        include_intercept: bool, default=True
            Should fitted model include an intercept or not
        """
        super().__init__()
        self.include_intercept_ = include_intercept
        self.solver_ = solver
        self.lam_ = lam
        self.penalty_ = penalty
        self.alpha_ = alpha

        if penalty not in ["none", "l1", "l2"]:
            raise ValueError("Supported penalty types are: none, l1, l2")

        self.coefs_ = None

    def _choose_regularization_module(self):
        """
        helper func to fit that  choose regularization module according to self.penalty_
        :param w: initial weights
        :return: choosen regularization module
        """
        PENALTY_DICT = {"none": 0, "l1": 1, "l2": 2}
        modules = []
        modules.append(LogisticModule(self.coefs_))
        modules.append(RegularizedModule(fidelity_module=LogisticModule(),
                 regularization_module=L1(),
                 lam=self.lam_,
                 weights=self.coefs_,
                 include_intercept=self.include_intercept_))
        modules.append(RegularizedModule(fidelity_module=LogisticModule(),
                 regularization_module=L2(),
                 lam=self.lam_,
                 weights=self.coefs_,
                 include_intercept=self.include_intercept_))
        return modules[PENALTY_DICT[self.penalty_]]

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit Logistic regression model to given samples
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for
        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        Notes
        -----
        Fits model using specified `self.optimizer_` passed when instantiating class and includes an intercept
        if specified by `self.include_intercept_
        """
        ones_to_concatenate = np.ones(X.shape[0])
        concatenated_X = np.r_['-1,2,0', ones_to_concatenate, X]
        if self.include_intercept_:
            X = concatenated_X

        self.coefs_ = np.random.randn(X.shape[1])
        logistic_module = LogisticModule()
        regularized_module = self._choose_regularization_module()
        self.coefs_ = self.solver_.fit(regularized_module, X, y)

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for
        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        predicted = np.where(self.predict_proba(X) > self.alpha_, 1, 0)
        return predicted


    def predict_proba(self, X: np.ndarray) -> np.ndarray:
            """
            Predict probabilities of samples being classified as `1` according to sigmoid(Xw)
            Parameters
            ----------
            X : ndarray of shape (n_samples, n_features)
                Input data to predict probability for
            Returns
            -------
            probabilities: ndarray of shape (n_samples,)
                Probability of each sample being classified as `1` according to the fitted model
            """
            ones_to_concatenate = np.ones(X.shape[0])
            concatenated_X = np.r_['-1,2,0', ones_to_concatenate, X]
            if self.include_intercept_:
                X = concatenated_X

            x_mult_w = np.matmul(X, self.coefs_)
            exp_x_mult_w = np.exp(-x_mult_w)
            sigmoid_x_mult_w = (1 / (exp_x_mult_w + 1))
            return sigmoid_x_mult_w


    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification error
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples
        y : ndarray of shape (n_samples, )
            True labels of test samples
        Returns
        -------
        loss : float
            Performance under misclassification error
        """
        from ...metrics import misclassification_error
        predicted_y = self._predict(X)
        return misclassification_error(y_true=y, y_pred=predicted_y, normalize=True)
