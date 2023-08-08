import numpy as np
import pandas as pd
from typing import Tuple, List, Callable, Type

from IMLearn import BaseModule
from IMLearn.desent_methods import GradientDescent, FixedLR, ExponentialLR
from IMLearn.desent_methods.modules import L1, L2
from IMLearn.learners.classifiers.logistic_regression import LogisticRegression
from IMLearn.utils import split_train_test
import plotly.graph_objects as go
from  IMLearn.model_selection import cross_validate


FUNCS_LIST = [L1,L2]
FUNCS_STRINGS = ['L1','L2']
ALPHA = 0.5
LAMDAS = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]


def plot_descent_path(module: Type[BaseModule],
                      descent_path: np.ndarray,
                      title: str = "",
                      xrange=(-1.5, 1.5),
                      yrange=(-1.5, 1.5)) -> go.Figure:
    """
    Plot the descent path of the gradient descent algorithm
    Parameters:
    -----------
    module: Type[BaseModule]
        Module type for which descent path is plotted
    descent_path: np.ndarray of shape (n_iterations, 2)
        Set of locations if 2D parameter space being the regularization path
    title: str, default=""
        Setting details to add to plot title
    xrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range
    yrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range
    Return:
    -------
    fig: go.Figure
        Plotly figure showing module's value in a grid of [xrange]x[yrange] over which regularization path is shown
    Example:
    --------
    fig = plot_descent_path(IMLearn.desent_methods.modules.L1, np.ndarray([[1,1],[0,0]]))
    fig.show()
    """
    def predict_(w):
        return np.array([module(weights=wi).compute_output() for wi in w])

    from utils import decision_surface
    return go.Figure([decision_surface(predict_, xrange=xrange, yrange=yrange, density=70, showscale=False),
                      go.Scatter(x=descent_path[:, 0], y=descent_path[:, 1], mode="markers+lines", marker_color="black")],
                     layout=go.Layout(xaxis=dict(range=xrange),
                                      yaxis=dict(range=yrange),
                                      title=f"GD Descent Path {title}"))


def get_gd_state_recorder_callback() -> Tuple[Callable[[], None], List[np.ndarray], List[np.ndarray]]:
    """
    Callback generator for the GradientDescent class, recording the objective's value and parameters at each iteration
    Return:
    -------
    callback: Callable[[], None]
        Callback function to be passed to the GradientDescent class, recoding the objective's value and parameters
        at each iteration of the algorithm
    values: List[np.ndarray]
        Recorded objective values
    weights: List[np.ndarray]
        Recorded parameters
    """
    values_list, weights_list = [],[]
    def callback(**kwargs):
        curr_weights = kwargs['weights']
        weights_list.append(curr_weights)

        curr_values = kwargs['val']
        values_list.append(curr_values)

    return callback, values_list, weights_list

def helper_compare_fixed_learning_rates(ind, init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                 etas: Tuple[float] = (1, .1, .01, .001)):
    """
    Same as compare_fixed_learning_rates, but get f to use:
    """
    X=None
    y=None
    lowest_val_list = []
    for eta in etas:
        my_callback, values_list, weights_list = get_gd_state_recorder_callback()
        gradientDescent = GradientDescent(learning_rate=FixedLR(eta),
                 tol=1e-5,
                 max_iter=1000,
                 out_type="best",
                 callback=my_callback)
        lowest_w = gradientDescent.fit(FUNCS_LIST[ind](np.copy(init)),X,y)
        lowest_val = np.abs(FUNCS_LIST[ind](lowest_w).compute_output())
        lowest_val_list.append(lowest_val)
        #Q1 : Plot the descent path for each of the settings described above for η = 0.01

        fig_Q_1 = plot_descent_path(FUNCS_LIST[ind],
                                    np.concatenate(weights_list,axis=0).reshape(len(weights_list),len(init)),
                                    "Descent path η = "+str(eta)+ " of norm" + FUNCS_STRINGS[ind])
        # print("Q1_eta_0.01_norm_" + FUNCS_STRINGS[ind]+".png")
        fig_Q_1.write_image("Q1_eta_"+str(eta)+"_norm_" + FUNCS_STRINGS[ind]+".png")

        # Q3: For each of the modules, plot the convergence rate (i.e. the norm as a function of the GD iteration)
        # for all specified learning rates
        norm = values_list
        GD_iterations = np.arange(len(norm))
        fig_Q_3 = go.Figure().add_traces(go.Scatter(x=GD_iterations, y=norm, mode="markers", showlegend=False))
        fig_Q_3.update_layout(title = "The " + FUNCS_STRINGS[ind] +"norm as a function of the GD iteration")
        fig_Q_3.write_image("Q_3_eta"+str(eta)+"_norm_" + FUNCS_STRINGS[ind]+".png")

        # Q4: The lowest loss achieved when minimizing each of the modules
    lowest_val_ind = np.argmin(lowest_val_list)
    print("The lowest loss achieved when minimizing norm " +
          FUNCS_STRINGS[ind] +" and η = "+str(etas[lowest_val_ind]) + " is :" + str(lowest_val_list[lowest_val_ind]))

def compare_fixed_learning_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                 etas: Tuple[float] = (1, .1, .01, .001)):
    for ind in range(len(FUNCS_LIST)):
        helper_compare_fixed_learning_rates(ind)




def compare_exponential_decay_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                    eta: float = .1,
                                    gammas: Tuple[float] = (.9, .95, .99, 1)):
    return
    # Optimize the L1 objective using different decay-rate values of the exponentially decaying learning rate
    raise NotImplementedError()

    # Plot algorithm's convergence for the different values of gamma
    raise NotImplementedError()

    # Plot descent path for gamma=0.95
    raise NotImplementedError()


def load_data(path: str = "../datasets/SAheart.data", train_portion: float = .8) -> \
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load South-Africa Heart Disease dataset and randomly split into a train- and test portion
    Parameters:
    -----------
    path: str, default= "../datasets/SAheart.data"
        Path to dataset
    train_portion: float, default=0.8
        Portion of dataset to use as a training set
    Return:
    -------
    train_X : DataFrame of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set
    train_y : Series of shape (ceil(train_proportion * n_samples), )
        Responses of training samples
    test_X : DataFrame of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set
    test_y : Series of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples
    """
    df = pd.read_csv(path)
    df.famhist = (df.famhist == 'Present').astype(int)
    return split_train_test(df.drop(['chd', 'row.names'], axis=1), df.chd, train_portion)


def compute_err(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate misclassification loss
    Returns
    -------
    Misclassification of given predictions
    """
    score = np.sum(y_true != y_pred)
    return score / len(y_true)

def fitting_chosen_norm_regularized_logistic_regression_models(chosen_norm,X_train, y_train, X_test, y_test):
    """
    Code for Q10 and Q11- Fitting l1- and l2-regularized logistic regression models,
    using cross-validation to specify values
    of regularization parameter
    :param chosen_norm: string represent chosen norm- l1 or l2
    :return:
    """
    # • Set α = 0.5:
    # • Use your previously implemented cross-validation procedure to choose λ:
    scores_list = []
    for lamda in LAMDAS:
        gradientDescent = GradientDescent(learning_rate=FixedLR(1e-4),
                                          tol=1e-5,
                                          max_iter=20000)
        logistic_regression = LogisticRegression(include_intercept=True,
             solver=gradientDescent,
             penalty=chosen_norm,
             lam=lamda,
             alpha=ALPHA)
        train_score, validation_score = cross_validate(logistic_regression,X_train,y_train,compute_err)
        scores_list.append(validation_score)

    best_lamda_ind = np.argmin(scores_list)
    best_lamda = LAMDAS[best_lamda_ind]
    gradientDescent = GradientDescent(learning_rate=FixedLR(1e-4),
                                      tol=1e-5,
                                      max_iter=20000)
    match_logistic_regression = LogisticRegression(include_intercept=True,
                                             solver=gradientDescent,
                                             penalty=chosen_norm,
                                             lam=best_lamda,
                                             alpha=ALPHA)
    # • After selecting λ repeat fitting with the chosen λ and α = 0.5 over the entire train portion
    match_logistic_regression.fit(X_train, y_train)
    y_predicted = match_logistic_regression.predict(X_test)
    test_error = compute_err(y_test, y_predicted)
    print("Fitting "+ chosen_norm + " regularized logistic regression model:")
    print("     (*) Value of λ was selected is: "+str(best_lamda))
    print("     (*)The model’s test error is: " + str(test_error))



def fit_logistic_regression():
    # Load and split SA Heard Disease dataset
    X_train, y_train, X_test, y_test = load_data()
    X_train, y_train, X_test, y_test = X_train.to_numpy(), y_train.to_numpy(), X_test.to_numpy(), y_test.to_numpy()

    # Plotting convergence rate of logistic regression over SA heart disease data
    #  Use max_iter=20,000 and lr=1e-4.
    gradientDescent = GradientDescent(learning_rate=FixedLR(1e-4),
                                      tol=1e-5,
                                      max_iter=20000)

    logistic_regression = LogisticRegression(gradientDescent)
    #Fit the model:
    logistic_regression.fit(X_train,y_train)

    # COPPIED FROM LAB 4:
    y_prob = logistic_regression.predict_proba(X_test)
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    fig_Q_8 = go.Figure(
        data=[go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="black", dash='dash'),
                         name="Random Class Assignment"),
              go.Scatter(x=fpr, y=tpr, mode='markers+lines', text=thresholds, name="", showlegend=False,
                         marker_size=5)],
        layout=go.Layout(title=rf"$\text{{ROC Curve Of Fitted Model - AUC}}={auc(fpr, tpr):.6f}$",
                         xaxis=dict(title=r"$\text{False Positive Rate (FPR)}$"),
                         yaxis=dict(title=r"$\text{True Positive Rate (TPR)}$")))

    # fig_Q_8.show()
    fig_Q_8.write_image("fig_Q_8.png")

    #Q9:
    roc = tpr - fpr
    ind_alpha_of_optimal_roc =  np.argmax(roc)
    alpha_of_optimal_roc = thresholds[ind_alpha_of_optimal_roc]
    print("Value of α achieves the optimal ROC value is: "+ str(alpha_of_optimal_roc))
    optimal_logistic_regression = LogisticRegression(alpha=alpha_of_optimal_roc)
    optimal_logistic_regression.fit(X_train,y_train)
    test_error = optimal_logistic_regression.loss(X_test, y_test)
    print("model’s test error of α achieves the optimal ROC value is: "+ str(test_error))

    # Fitting l1- and l2-regularized logistic regression models, using cross-validation to specify values
    # of regularization parameter

    # Q10: ℓ1-regularized logistic regression
    fitting_chosen_norm_regularized_logistic_regression_models("l1",X_train, y_train, X_test, y_test)
    # Q11: ℓ2-regularized logistic regression
    fitting_chosen_norm_regularized_logistic_regression_models("l2",X_train, y_train, X_test, y_test)


if __name__ == '__main__':
    np.random.seed(0)
    compare_fixed_learning_rates()
    compare_exponential_decay_rates()
    fit_logistic_regression()
