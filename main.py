from model import Model
from optimizer import OptimizerFactory

dim = 10
model = Model(dim = dim)
model.set_objective(name = 'quad_fun1')
model.diff()
model.diff_second()

n_iter, eps = 200, 10**(-3)
optimizer_factory = OptimizerFactory(model = model, eps = eps)
optimizer = optimizer_factory.create_method(method = 'BFGS')
optimizer.set_init(lb = 0, ub = 100)

for k in range(n_iter):
    optimizer.search_direction(k)
    if optimizer.is_stop():
        break
    optimizer.get_step()
    optimizer.update()
    optimizer.moniter(iter_time = k)
    

