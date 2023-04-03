# Testing-Nonlinear-Least-Squares-and-Neural-Networks-Parameters
The objective was to implement Nonlinear Least Squares (NLS) for GPS localization
using various initial values and to evaluate a neural networks performance for plant classification
using different learning rates. For NLS the objective function was evaluated using ‘lsqnonlin’. The
input to ‘lsqnonlin’ consists of the objective function and three different initial values: [14131454.8,
13301808.7, 17048044.1] (W0), 0.5*W0, and 2*W0. For each unique initial value, a different
Cartesian coordinate of the receiver was found. This displays that the evaluated objective function
has multiple minimizers and that the initial value has an effect on which minimizer is found. The
neural network was run with learning rates of 0.01, 1, and 0.000001 which attained an accuracy of
96%, 0.6667%, and 36% respectively. As displayed by the results, the learning rate has a strong
effect on the accuracy of the model in question. A strong initial estimate for NLS problems is
important, as it is possible to get vastly different results based on this initial value. Similarly for
neural networks, the learning rate has a noticeable effect on the performance of the model.
Implementing a method such as ADADELTA is a good idea, as it requires no manual tuning of the
learning rate [1].
