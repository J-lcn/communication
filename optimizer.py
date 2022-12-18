#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from line_search import armijo
from line_search import exact
from line_search import constant


# In[2]:


class OptimizerFactory:
    def __init__(self, model, eps):
        self.model, self.eps = model, eps
        self.method_dict = {'Gradient': Gradient, 'Newton': Newton , 'ModifiedNewton': ModifiedNetwon,'BFGS':BFGS, 'DFP': DFP}
    
    def create_method(self, method):
        return self.method_dict.get(method)(self.model, self.eps)


# In[3]:


class Optimizer:
    def __init__(self, model, eps) -> None:
        self.m, self.eps = model, eps
        self.trace_obj, self.trace_x = [], []
        self.trace_step, self.trace_d = [], []
    
    def set_init(self, lb = 0, ub = 1)-> None:
        assert(lb <= ub), print("ub is larger than lb")
        self.x = (ub - lb) + np.random.rand(self.m.dim) + lb * np.ones(self.m.dim)
    
    def first_order(self):
        self.grad, self.obj_values = self.m.diff_eval(self.x), self.m.obj_fun_eval(self.x)
        return self.grad, self.obj_values
    
    def second_order(self):
        self.hessian = self.m.diff_second_eval(self.x)
        return self.hessian
    
    def get_step(self):
        # self.step = armijo(self.x, self.d, self.grad, self.obj_values, self.m.obj_fun_eval)
        self.step = exact(self.x, self.d, self.m.obj_fun_eval)

    def moniter(self, iter_time):
        self.trace_obj.append(self.obj_values)
        self.trace_x.append(self.x)
        self.trace_d.append(self.d)
        self.trace_step.append(self.step)
        if iter_time % 20 == 19:
            print("iterations {} - objective function: {} - grad norm: {}".format(iter_time, self.obj_values, np.linalg.norm(self.grad)))
    
    def is_stop(self):
        if np.linalg.norm(self.grad, 1.0) <= self.eps:
            return True
        return False
    
    def update(self):
        assert(self.step != False), print("objective function is not decreased")
        self.x += self.step * self.d
    
    def save(self):
        self.file_route = "./results//"
        pd.DataFrame(self.trace_obj).to_csv(self.file_route + "trace_obj.csv")
        pd.DataFrame(self.trace_x).to_csv(self.file_route + "trace_x.csv")
        pd.DataFrame(self.trace_d).to_csv(self.file_route + "trace_d.csv")
        pd.DataFrame(self.trace_step).to_csv(self.file_route + "trace_step.csv")


# In[4]:


class Newton(Optimizer):
    def __init__(self, model, eps) -> None:
        super().__init__(model, eps)
        
        
    def get_step(self):
        # self.step = armijo(self.x, self.d, self.grad, self.obj_values, self.m.obj_fun_eval)
        self.step = 1
    
    def search_direction(self, iter_time):
        self.grad, self.obj_values = self.first_order()
        self.hessian = self.second_order()
        self.d = -1.0 * np.dot(np.linalg.inv(self.hessian), self.grad)
        self.iter_time=iter_time


# In[5]:


class ModifiedNetwon(Optimizer):
    def __init__(self, model, eps) -> None:
        super().__init__(model, eps)
    
    def search_direction(self, miu,iter_time):
        self.grad, self.obj_values = self.first_order()
        self.hessian = self.second_order()
        self.d = -1.0 * np.dot(np.linalg.inv(self.hessian + miu*np.eye(self.hessian.shape[0])), self.grad)


# In[6]:


class Gradient(Optimizer):
    def __init__(self, model, eps) -> None:
        super().__init__(model, eps)
    
    def search_direction(self):
        self.grad, self.obj_values = self.first_order()
        self.d = -1.0 * self.grad


# In[7]:


class BFGS(Optimizer):
    def __init__(self, model, eps) -> None:
        super().__init__(model, eps)
        self.h = self.__generate_approx_hessian()
    
    def __generate_approx_hessian(self):
        return np.eye(self.m.dim)
    
    def __approx_hessian(self):
        self.s, self.y = self.x - self.x_hat, self.grad - self.grad_hat
        self.ys = np.dot(self.y.T, self.s)
        self.hy = np.dot(self.h, self.y)
        self.h += (1.0/self.ys) *(1 + np.dot(self.y.T, self.hy)/self.ys)*np.dot(self.s, self.s.T) \
        - (1.0/self.ys)*(np.dot(np.dot(self.s, self.y.T), self.h) + np.dot(self.hy, self.s.T))
        return self.h
    
    def search_direction(self, iter_time):
        self.grad, self.obj_values = self.first_order()
        if iter_time == 0:
            self.d = -1.0 * self.grad
        else:
            self.h = self.__approx_hessian()
            self.d = -1.0 * np.dot(self.h, self.grad)
        self.grad_hat, self.x_hat = self.grad.copy(), self.x.copy()


# In[8]:


class DFP(Optimizer):
    def __init__(self, model, eps) -> None:
        super().__init__(model, eps)
        self.h = self.__generate_approx_hessian()
    
    def __generate_approx_hessian(self):
        return np.eye(self.m.dim)
    
    def __approx_hessian(self):
        self.s, self.y = self.x - self.x_hat, self.grad - self.grad_hat
        self.sy = np.dot(self.s.T, self.y)
        self.hy = np.dot(self.h, self.y)
        self.h += (1/self.sy) * np.dot(self.s, self.s.T) - np.dot(np.dot(self.hy, self.y.T), self.h) / (np.dot(self.y.T, self.hy))
    
    def search_direction(self, iter_time):
        self.grad, self.obj_values = self.first_order()
        if iter_time == 0:
            self.d = -1.0 * self.grad
        else:
            self.h = self.__approx_hessian()
            self.d = -1.0 * np.dot(self.h, self.grad)
        self.grad_hat, self.x_hat = self.grad.copy(), self.x.copy()



# In[ ]:




