"""Hyperboloid manifold."""
'''
问题：
1. inner内积函数问题: 为什么需要符号调整操作：dp.narrow(-1, 0, 1).mul_(-1)
在双曲神经网络的梯度转换时，将切空间的梯度转换到双曲空间时，为什么要对切空间的梯度的时间分量，即第一个分量，变换符号呢？−x02​+x12​+x22​+⋯+xd2​
'''
import torch
import inspect
from manifolds.base import Manifold
from utils.math_utils import arcosh, cosh, sinh 


class Hyperboloidtest(Manifold):
    """
    这里使用的论文2的方法，基于Minkowski 内积计算距离
    sqdist(x,y,c)=K⋅(arcosh(−⟨x,y⟩/K​))2
    Hyperboloid manifold class.
    必须要有的egrad2rgrad、init_weights 、inner、proj(已有)、ptransp(已有)、proj_tan(已有)、sqdist(已有)、expmap(已有)
    不同于poincareball，这里没有缩放因子_lambda_x
    c = 1 / K is the hyperbolic curvature. 
    """

    def __init__(self):
        super(Hyperboloidtest, self).__init__()
        self.name = 'Hyperboloid'
        self.eps = {torch.float32: 1e-7, torch.float64: 1e-15}
        self.min_norm = 1e-15
        self.max_norm = 1e6

    def inner(self, p, c, u, v=None, keepdim = False, dim = -1):
        """
        计算洛伦兹内积
        参数:
            p: 流形内的点，形状为 (..., d+1)
            c: 曲率 (暂不使用，但保留以匹配函数签名)
            u: 切向量，形状为 (..., d+1)
            v: 第二个切向量，形状为 (..., d+1)。如果为 None，则计算 u 的自内积，
            u 是切向量，不需要满足这个hyperboloid的约束要求，而是被定义在点 pp 的切空间内
            keepdim: 是否保持维度

        返回:
            洛伦兹内积，形状取决于 keepdim
        """
        if v is None:
            v=u
        d = u.size(dim) - 1
        uv = u * v
        if keepdim is False:
            return -uv.narrow(dim, 0, 1).sum(dim=dim, keepdim=False) + uv.narrow(dim, 1, d).sum(dim=dim, keepdim=False)
        else:
            return torch.cat((-uv.narrow(dim, 0, 1), uv.narrow(dim, 1, d)), dim=dim).sum(dim=dim, keepdim=True)
        
    def egrad2rgrad(self, p, dp, c):
        """
        论文2，公式10下面的公式
        将欧氏梯度转换为黎曼梯度
        黎曼梯度是 欧几里得梯度dp 在位于流形中的点p的切空间中的正交投影
        参数:
            p: 流形内的点，形状为 (..., d+1)
            dp: 欧氏梯度，形状为 (..., d+1)
            c: 曲率 (暂不使用，但保留以匹配函数签名)

        返回:
            黎曼梯度，形状为 (..., d+1)
        """
        #minkowski_norm_p = self.minkowski_norm(p, keepdim=True)#这个值算出来应该等于-k或-1/c，但是minkowski_norm无法满足要求，所以不能用minkowski_norm计算自内积
        dp.narrow(-1, 0, 1).mul_(-1)#hyperboloid的点的约束条件，要求时间分量的平方取符号，因此求导后也要去负号。
        inner_prod = self.inner(p=p,c=c,u=p,v=dp, keepdim=True)  # <dp, p>_L
        #rgrad = dp - inner_prod / minkowski_norm_p.clamp(min=self.min_norm) * p
        rgrad = dp + inner_prod * p * c
        return rgrad #torch.clamp(rgrad, min=self.min_norm) # 根据公式转换

    def init_weights(self, w, c, irange=0.001):
        '''
        在流形上随机初始化节点嵌入。
        根据方程 (6) 初始化：
            x' 从 U(-0.001, 0.001) 采样
            x0 = sqrt(1 + ||x'||^2)
        
        参数:
            w: 要初始化的权重张量，形状为 (batch_size, d+1)
            c: 曲率参数
            irange: 初始化范围，用于均匀分布 U(-irange, irange)
        
        返回:
            初始化后的权重张量，确保所有点位于流形上
        '''
        K = 1. / c  # 根据方程 (6)，假设 K = 1
        d = w.size(-1) - 1  # 空间维度
        
        # 1. 初始化空间分量 x' 从 U(-0.001, 0.001) 采样
        w[:, 1:].data.uniform_(-irange, irange)
        
        # 2. 计算 x0 = sqrt(1 + ||x'||^2)
        y_sqnorm = torch.sum(w[:, 1:] ** 2, dim=-1, keepdim=True)
        w[:, 0:1].data = torch.sqrt(K + y_sqnorm.clamp(min=self.min_norm))
        
        # 3. 投影到流形上（确保数值稳定性）
        w = self.proj(w, c)
        print(f"初始嵌入W:",w)
        # 4. 验证初始化是否正确
        with torch.no_grad():
            # 检查是否有 NaN 或 Inf
            if torch.isnan(w).any():
                raise ValueError("初始化后的权重包含 NaN 值。")
            if torch.isinf(w).any():
                raise ValueError("初始化后的权重包含 Inf 值。")
        return w
    '''上面的函数是新添加的'''

    def minkowski_dot(self, x, y, keepdim=True):
        '''
        minkowski 内积，它是 伪欧几里得内积,相当于poincare中的inner函数
        x, y: 张量，形状为 (..., d+1)
        keepdim: 是否保持维度
        返回值: 张量，形状为 (...) 或 (..., 1) 取决于 keepdim
        '''
        res = torch.sum(x * y, dim=-1) - 2 * x[..., 0] * y[..., 0]
        #res = -x[..., 0] * y[..., 0] + torch.sum(x[..., 1:] * y[..., 1:], dim=-1)
        if keepdim:
            res = res.view(res.shape + (1,))
        return res
    
    def minkowski_norm(self, u, keepdim=True):
        """
        计算洛伦兹范数
        u: 张量，形状为 (..., d+1)
        keepdim: 是否保持维度
        返回值: 张量，形状为 (...) 或 (..., 1) 取决于 keepdim
        双曲空间的Minkowski 内积是负数，这里会对负数取平方根，会报错
        """
        # 获取调用者的函数名称
        caller_name = inspect.stack()[1].function
        #print(f"Calling function: {caller_name}")
        dot = self.minkowski_dot(u, u, keepdim=keepdim)
        #print(f"dot:", dot)
        return torch.sqrt(torch.clamp(dot, min=self.eps[u.dtype]))  # Minkowski 内积是负数，这里会对负数取平方根，会报错
    
    def sqdist(self, x, y, c):
        """Squared distance between pairs of points.
        洛伦兹平方距离
        """
        K = 1. / c
        prod = self.minkowski_dot(x, y)#计算 x 和 y 在双曲空间中的 Minkowski内积⟨x,y⟩
        theta = torch.clamp(-prod / K, min=1.0 + self.eps[x.dtype])#双曲距离度量中的夹角参数 θ,θ=−⟨x,y⟩​/K
        sqdist = K * arcosh(theta) ** 2#用 反双曲余弦函数（arcosh）来计算 双曲空间中的距离，论文2
        #sqdist = - 2.0 * K - 2.0 * self.inner(p=x, c=c, u=x, v=y,keepdim = True)#论文3
        #sqdist = - 2.0 * K - 2.0 * self.minkowski_dot(x, y)#论文3
        sqdist = torch.clamp(sqdist,min=self.min_norm)
        # clamp distance to avoid nans in Fermi-Dirac decoder
        u=x
        v=y
        u0 = torch.sqrt(torch.sum(torch.pow(u,2),dim=-1, keepdim=True) + K)#时间分量，
        v0 = -torch.sqrt(torch.sum(torch.pow(v,2),dim=-1, keepdim=True) + K)#时间分量
        u = torch.cat((u,u0),dim=-1)
        v = torch.cat((v,v0),dim=-1)
        #sqdist = - 2 * K - 2 *torch.sum(u * v, dim=-1, keepdim=True)#论文3的公式6
        return torch.clamp(sqdist, max=50.0)#sqdist 的值会被裁剪，确保它不会超过 50.0

    def proj(self, x, c):
        """Projects point x on the manifold.
        将点x投影到流形上,确保投影后的点位于流形上，即满足流形的约束条件
        """
        K = 1. / c
        d = x.size(-1) - 1
        y = x.narrow(-1, 1, d)#提取x的空间分量(x1​,x2​,…,xd​)，忽略了时间分量x0​。
        y_sqnorm = torch.norm(y, p=2, dim=1, keepdim=True) ** 2 #计算空间分量(x1​,x2​,…,xd​)欧几里得平方范数
        mask = torch.ones_like(x)
        mask[:, 0] = 0#mask用于分离时间分量x0和空间分量(x1​,x2​,…,xd​)
        vals = torch.zeros_like(x)
        vals[:, 0:1] = torch.sqrt(torch.clamp(K + y_sqnorm, min=self.eps[x.dtype]))
        '''
        vals为计算时间分量的投影值：x0=(K+sum(x1​,x2​,…,xd​)**2)**0.5，
        这样得到的-x0**2+(x1​,x2​,…,xd​)**2=-K，确保在流形上
        '''
        return vals + mask * x

    def proj_tan(self, u, x, c):
        """
        Projects u on the tangent space of x.
        将一个d+1维的向量 u 投影到d+1维的 x 的切空间上
        将一个向量 u 投影到x的切空间（tangent space）
        """
        K = 1. / c
        d = x.size(1) - 1# 获取的是 x 第一个维度上的大小-1
        ux = torch.sum(x.narrow(-1, 1, d) * u.narrow(-1, 1, d), dim=1, keepdim=True)#进行u和x的点积计算
        #x.narrow(-1, 1, d)在最后一个维度（即坐标维度）上从索引 1 开始，选取长度为 d 的子向量，
        #即点 x 的第一维（时间维度）去掉，只保留空间维度的部分，再x和u相乘，再求和：进行点积计算
        mask = torch.ones_like(u)
        mask[:, 0] = 0
        vals = torch.zeros_like(u)
        vals[:, 0:1] = ux / torch.clamp(x[:, 0:1], min=self.eps[x.dtype])#根据双曲几何中的切空间投影公式，计算时间坐标上的分量
        return vals + mask * u

    def proj_tan0(self, u, c):
        narrowed = u.narrow(-1, 0, 1)
        vals = torch.zeros_like(u)
        vals[:, 0:1] = narrowed
        return u - vals

    def expmap(self, u, x, c):
        '''
        论文2的公式9
        位于点x的切向量u向双曲模型映射
        切向量u不需要满足约束条件
        '''
        K = 1. / c
        sqrtK = K ** 0.5
        print(f"expmap.u:",u)
        normu = self.minkowski_norm(u)
        normu = torch.clamp(normu, max=self.max_norm)
        theta = normu / sqrtK
        theta = torch.clamp(theta, min=self.min_norm)
        result = cosh(theta) * x + sinh(theta) * u / theta
        return self.proj(result, c)
        
    def logmap(self, x, y, c):
        K = 1. / c
        xy = torch.clamp(self.minkowski_dot(x, y) + K, max=-self.eps[x.dtype]) - K
        u = y + xy * x * c
        normu = self.minkowski_norm(u)
        normu = torch.clamp(normu, min=self.min_norm)
        dist = self.sqdist(x, y, c) ** 0.5
        result = dist * u / normu*2.0
        #myresult = dist * u /sinh(dist)
        #result = myresult
        return self.proj_tan(result, x, c)

    def expmap0(self, u, c):
        K = 1. / c
        sqrtK = K ** 0.5
        d = u.size(-1) - 1
        x = u.narrow(-1, 1, d).view(-1, d)
        x_norm = torch.norm(x, p=2, dim=1, keepdim=True)
        x_norm = torch.clamp(x_norm, min=self.min_norm)
        theta = x_norm / sqrtK
        res = torch.ones_like(u)
        res[:, 0:1] = sqrtK * cosh(theta)
        res[:, 1:] = sqrtK * sinh(theta) * x / x_norm
        return self.proj(res, c)

    def logmap0(self, x, c):
        K = 1. / c
        sqrtK = K ** 0.5
        d = x.size(-1) - 1
        y = x.narrow(-1, 1, d).view(-1, d)
        y_norm = torch.norm(y, p=2, dim=1, keepdim=True)
        y_norm = torch.clamp(y_norm, min=self.min_norm)
        res = torch.zeros_like(x)
        theta = torch.clamp(x[:, 0:1] / sqrtK, min=1.0 + self.eps[x.dtype])
        res[:, 1:] = sqrtK * arcosh(theta) * y / y_norm
        return res

    def mobius_add(self, x, y, c):
        u = self.logmap0(y, c)
        v = self.ptransp0(x, u, c)
        return self.expmap(v, x, c)

    def mobius_matvec(self, m, x, c):
        u = self.logmap0(x, c)
        mu = u @ m.transpose(-1, -2)
        return self.expmap0(mu, c)

    def ptransp(self, x, y, u, c):
        logxy = self.logmap(x, y, c)
        logyx = self.logmap(y, x, c)
        sqdist = torch.clamp(self.sqdist(x, y, c), min=self.min_norm)
        alpha = self.minkowski_dot(logxy, u) / sqdist
        res = u - alpha * (logxy + logyx)
        return self.proj_tan(res, y, c)

    def ptransp0(self, x, u, c):
        K = 1. / c
        sqrtK = K ** 0.5
        x0 = x.narrow(-1, 0, 1)
        d = x.size(-1) - 1
        y = x.narrow(-1, 1, d)
        y_norm = torch.clamp(torch.norm(y, p=2, dim=1, keepdim=True), min=self.min_norm)
        y_normalized = y / y_norm
        v = torch.ones_like(x)
        v[:, 0:1] = - y_norm 
        v[:, 1:] = (sqrtK - x0) * y_normalized
        alpha = torch.sum(y_normalized * u[:, 1:], dim=1, keepdim=True) / sqrtK
        res = u - alpha * v
        return self.proj_tan(res, x, c)

    def to_poincare(self, x, c):
        K = 1. / c
        sqrtK = K ** 0.5
        d = x.size(-1) - 1
        return sqrtK * x.narrow(-1, 1, d) / (x[:, 0:1] + sqrtK)