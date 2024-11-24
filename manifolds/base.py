"""Base manifold."""

from torch.nn import Parameter


class Manifold(object):
    """
    Abstract class to define operations on a manifold.
    """

    def __init__(self):
        super().__init__()
        self.eps = 10e-8

    def sqdist(self, p1, p2, c):
        """Squared distance between pairs of points."""
        raise NotImplementedError

    def egrad2rgrad(self, p, dp, c):
        """Converts Euclidean Gradient to Riemannian Gradients."""
        raise NotImplementedError

    def proj(self, p, c):
        """Projects point p on the manifold."""
        raise NotImplementedError

    def proj_tan(self, u, p, c):
        """Projects u on the tangent space of p."""
        raise NotImplementedError

    def proj_tan0(self, u, c):
        """Projects u on the tangent space of the origin."""
        raise NotImplementedError

    def expmap(self, u, p, c):
        """Exponential map of u at point p."""
        raise NotImplementedError

    def logmap(self, p1, p2, c):
        """Logarithmic map of point p1 at point p2."""
        raise NotImplementedError

    def expmap0(self, u, c):
        """Exponential map of u at the origin."""
        raise NotImplementedError

    def logmap0(self, p, c):
        """Logarithmic map of point p at the origin."""
        raise NotImplementedError

    def mobius_add(self, x, y, c, dim=-1):
        """Adds points x and y."""
        raise NotImplementedError

    def mobius_matvec(self, m, x, c):
        """Performs hyperboic martrix-vector multiplication."""
        raise NotImplementedError

    def init_weights(self, w, c, irange=1e-5):
        """Initializes random weigths on the manifold."""
        raise NotImplementedError

    def inner(self, p, c, u, v=None, keepdim=False):
        """Inner product for tangent vectors at point x."""
        raise NotImplementedError

    def ptransp(self, x, y, u, c):
        """Parallel transport of u from x to y."""
        raise NotImplementedError

    def ptransp0(self, x, u, c):
        """Parallel transport of u from the origin to y."""
        raise NotImplementedError


class ManifoldParameter(Parameter):#继承自 torch.nn.Parameter
    """
    Subclass of torch.nn.Parameter for Riemannian optimization.
    通过 __new__ 和 __init__ 方法将流形参数 data 包装成 PyTorch Parameter，
    同时还保存了流形（manifold）和曲率（c）等信息。
    这样在优化过程中，ManifoldParameter 不仅保存了模型参数，还会将流形的几何信息传递给优化器
    ManifoldParameter 在优化过程中将流形的几何操作（如梯度转换、投影等）与 PyTorch 的 Parameter 结合起来
    """
    def __new__(cls, data, requires_grad, manifold, c):
        return Parameter.__new__(cls, data, requires_grad)#调用父类构造函数，将 data（参数值）封装为张量，同时附加流形和曲率信息：

    def __init__(self, data, requires_grad, manifold, c):
        self.c = c# 流形的曲率
        self.manifold = manifold

    def __repr__(self):
        return '{} Parameter containing:\n'.format(self.manifold.name) + super(Parameter, self).__repr__()
