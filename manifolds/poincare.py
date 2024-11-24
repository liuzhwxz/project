"""Poincare ball manifold."""

import torch

from manifolds.base import Manifold
from utils.math_utils import artanh, tanh


class PoincareBall(Manifold):
    """
    这里最终的公式类似于论文1的方法，不过使用的是双曲正切的反函数artanh计算最终的距离
    PoicareBall Manifold class.

    We use the following convention: x0^2 + x1^2 + ... + xd^2 < 1 / c

    Note that 1/sqrt(c) is the Poincare ball radius.

    """

    def __init__(self, ):
        super(PoincareBall, self).__init__()
        self.name = 'PoincareBall'
        self.min_norm = 1e-15# 防止数值误差导致除零
        self.eps = {torch.float32: 4e-3, torch.float64: 1e-5}# 流形的数值精度控制

    def sqdist(self, p1, p2, c):
        '''
        sqdist 方法计算的平方距离：
        '''
        sqrt_c = c ** 0.5
        dist_c = artanh(
            sqrt_c * self.mobius_add(-p1, p2, c, dim=-1).norm(dim=-1, p=2, keepdim=False)
        )#计算
        dist = dist_c * 2 / sqrt_c
        return dist ** 2

    def _lambda_x(self, x, c):
        '''
        论文1的公式1上面的gx，Poincaré Ball 模型的黎曼度规张量Riemannian metric tensor中的共形因子（conformal factor），即λc​(x)
        x表示poincareball中的点
        通过共形因子（conformal factor），即λc​(x)，调整欧氏内积得到Poincaré 球模型中内积
        随着点 xx 离原点越来越远，度量的缩放因子会变小，从而导致空间的 弯曲
        '''
        x_sqnorm = torch.sum(x.data.pow(2), dim=-1, keepdim=True)
        return 2 / (1. - c * x_sqnorm).clamp_min(self.min_norm)

    def egrad2rgrad(self, p, dp, c):
        '''
        #梯度转换
        论文1的公式5
        将欧几里得梯度转换为 Riemann 梯度，在欧几里得空间中计算的梯度不能直接应用于流形上的优化
        '''
        lambda_p = self._lambda_x(p, c)
        dp /= lambda_p.pow(2)
        return dp

    def proj(self, x, c):
        '''
        论文1的公式5的上一个公式
        确保点始终位于流形内
        #投影（Projection,使用 proj 方法将嵌入投影回流形内，确保嵌入始终满足 Poincaré Ball 的约束条件：
        1.计算点 x 的欧几里得范数 norm。
        2.确定允许的最大范数 maxnorm。
        3.对超出范围的点进行缩放，使其位于球内。
        '''
        norm = torch.clamp_min(x.norm(dim=-1, keepdim=True, p=2), self.min_norm)
        maxnorm = (1 - self.eps[x.dtype]) / (c ** 0.5)
        cond = norm > maxnorm
        projected = x / norm * maxnorm
        return torch.where(cond, projected, x)

    def proj_tan(self, u, p, c):
        """Projects u on the tangent space of p.
        Poincaré Ball 流形的切空间本身就是标准的欧几里得空间，所以我们可以认为 所有向量已经在切空间内，无需额外的投影操作
        """
        return u

    def proj_tan0(self, u, c):
        return u

    def expmap(self, u, p, c):
        '''
        指数映射（expmap）将一个切向量从某一点映射到流形上的另一点
        在优化器中，指数映射用来根据给定的更新方向沿着切空间的方向更新模型参数。
        '''
        sqrt_c = c ** 0.5
        u_norm = u.norm(dim=-1, p=2, keepdim=True).clamp_min(self.min_norm)
        second_term = (
                tanh(sqrt_c / 2 * self._lambda_x(p, c) * u_norm)
                * u
                / (sqrt_c * u_norm)
        )
        gamma_1 = self.mobius_add(p, second_term, c)
        return gamma_1

    def logmap(self, p1, p2, c):
        '''
        对数映射（Log Map）
        将流形上的点 p2 映射到切空间中的向量。
        '''
        sub = self.mobius_add(-p1, p2, c)
        sub_norm = sub.norm(dim=-1, p=2, keepdim=True).clamp_min(self.min_norm)
        lam = self._lambda_x(p1, c)
        sqrt_c = c ** 0.5
        return 2 / sqrt_c / lam * artanh(sqrt_c * sub_norm) * sub / sub_norm

    def expmap0(self, u, c):
        sqrt_c = c ** 0.5
        u_norm = torch.clamp_min(u.norm(dim=-1, p=2, keepdim=True), self.min_norm)
        gamma_1 = tanh(sqrt_c * u_norm) * u / (sqrt_c * u_norm)
        return gamma_1

    def logmap0(self, p, c):
        sqrt_c = c ** 0.5
        p_norm = p.norm(dim=-1, p=2, keepdim=True).clamp_min(self.min_norm)
        scale = 1. / sqrt_c * artanh(sqrt_c * p_norm) / p_norm
        return scale * p

    def mobius_add(self, x, y, c, dim=-1):
        '''
        Möbius 加法
        
        分别计算分子 num 和分母 denom。
        num = (1+2c*⟨x,y⟩+c∥y∥2)*x+(1−c∥x∥2)*y
        denom = 1+2c*⟨x,y⟩+c2*∥x∥2*∥y∥2
        通过分子分母的比值实现加法。
        '''
        x2 = x.pow(2).sum(dim=dim, keepdim=True)
        y2 = y.pow(2).sum(dim=dim, keepdim=True)
        xy = (x * y).sum(dim=dim, keepdim=True)
        num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
        denom = 1 + 2 * c * xy + c ** 2 * x2 * y2
        return num / denom.clamp_min(self.min_norm)

    def mobius_matvec(self, m, x, c):
        sqrt_c = c ** 0.5
        x_norm = x.norm(dim=-1, keepdim=True, p=2).clamp_min(self.min_norm)
        mx = x @ m.transpose(-1, -2)
        mx_norm = mx.norm(dim=-1, keepdim=True, p=2).clamp_min(self.min_norm)
        res_c = tanh(mx_norm / x_norm * artanh(sqrt_c * x_norm)) * mx / (mx_norm * sqrt_c)
        cond = (mx == 0).prod(-1, keepdim=True, dtype=torch.uint8)
        res_0 = torch.zeros(1, dtype=res_c.dtype, device=res_c.device)
        res = torch.where(cond, res_0, res_c)
        return res

    def init_weights(self, w, c, irange=1e-5):
        '''
        在流形上随机初始化节点嵌入
        '''
        w.data.uniform_(-irange, irange)
        return w#这里无需再向proj投影，因为在irange初始化矩阵后，肯定在poincare ball流形内

    def _gyration(self, u, v, w, c, dim: int = -1):
        u2 = u.pow(2).sum(dim=dim, keepdim=True)
        v2 = v.pow(2).sum(dim=dim, keepdim=True)
        uv = (u * v).sum(dim=dim, keepdim=True)
        uw = (u * w).sum(dim=dim, keepdim=True)
        vw = (v * w).sum(dim=dim, keepdim=True)
        c2 = c ** 2
        a = -c2 * uw * v2 + c * vw + 2 * c2 * uv * vw
        b = -c2 * vw * u2 - c * uw
        d = 1 + 2 * c * uv + c2 * u2 * v2
        return w + 2 * (a * u + b * v) / d.clamp_min(self.min_norm)

    def inner(self, x, c, u, v=None, keepdim=False):
        '''
        x表示流形内的点，c为曲率，u、v表示切向量
        Poincaré 球模型中内积，
        通过共形因子（conformal factor），即λc​(x)，调整欧氏内积得到Poincaré 球模型中内积
        '''
        if v is None:
            v = u
        lambda_x = self._lambda_x(x, c)
        return lambda_x ** 2 * (u * v).sum(dim=-1, keepdim=keepdim)

    def ptransp(self, x, y, u, c):
        '''
        切向量平行移动
        将切空间上的向量 uu 从点 xx 平移到点 yy。
        '''
        lambda_x = self._lambda_x(x, c)
        lambda_y = self._lambda_x(y, c)
        return self._gyration(y, -x, u, c) * lambda_x / lambda_y

    def ptransp_(self, x, y, u, c):
        lambda_x = self._lambda_x(x, c)
        lambda_y = self._lambda_x(y, c)
        return self._gyration(y, -x, u, c) * lambda_x / lambda_y

    def ptransp0(self, x, u, c):
        lambda_x = self._lambda_x(x, c)
        return 2 * u / lambda_x.clamp_min(self.min_norm)

    def to_hyperboloid(self, x, c):
        #论文2中公式11
        K = 1./ c
        sqrtK = K ** 0.5
        sqnorm = torch.norm(x, p=2, dim=1, keepdim=True) ** 2# x在最后一个维度的平方范数，即dim=1 表示计算每一行（假设 x 是一个二维张量）的范数，keepdim=True 表示保持输出的形状为 (节点数, 1)
        return sqrtK * torch.cat([K + sqnorm, 2 * sqrtK * x], dim=1) / (K - sqnorm)

