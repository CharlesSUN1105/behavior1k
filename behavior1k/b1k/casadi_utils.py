import casadi as cs

INFTY = 1e12  # better to use a big number


def vectorize(x: list[cs.SX]):
    return cs.vertcat(*[cs.vec(el) for el in x])


def zeros_like(x: cs.DM | cs.SX):
    return cs.DM.zeros(*x.shape)


def ones_like(x: cs.DM | cs.SX):
    return cs.DM.ones(*x.shape)
