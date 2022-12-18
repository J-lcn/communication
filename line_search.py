import numpy as np

def armijo(x, d, grad, obj_values, obj_fun_eval):
    alpha, beta, pho = 10, 0.1, 0.8
    for _ in range(100):
        if obj_fun_eval(x + alpha*d) <= obj_values + alpha * pho * np.dot(grad.T, d):
            return alpha
        alpha *= beta
    return False

def exact(x, d, obj_fun_eval):
    alpha = np.linspace(10, 0.001, 100)
    index = np.argmin([obj_fun_eval(x + a*d) for a in alpha])
    return alpha[index]

def constant():
    return 1.0
     