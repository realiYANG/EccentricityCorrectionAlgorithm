# 导入所需的库
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt


# 定义一个函数，根据圆心和半径计算极坐标下的点到圆的距离
def dist_to_circle(c, r, theta):
    # c是圆心的极坐标，r是半径，theta是点的角度
    # 返回一个数组，表示每个点到圆的距离
    return np.abs(c[0] + r * np.cos(theta - c[1]) - r)


# 定义一个函数，根据圆心和半径计算直角坐标下的圆的方程
def circle_equation(c, r, x):
    # c是圆心的直角坐标，r是半径，x是横坐标
    # 返回一个数组，表示每个横坐标对应的纵坐标
    return np.sqrt(np.maximum(r ** 2 - (x - c[0]) ** 2, 0)) + c[1]


# 生成一些模拟数据，假设圆心为(0, 0)，半径为1，加上一些噪声
c_true = (0, 0)  # 真实的圆心
r_true = 1  # 真实的半径
n = 20  # 数据点的个数
theta = np.linspace(0, 2 * np.pi, n)  # 生成均匀分布的角度
rho = c_true[0] + r_true * np.cos(theta - c_true[1]) + np.random.normal(0, 0.01, n)  # 生成带噪声的极径
x = rho * np.cos(theta)  # 转换为直角坐标的横坐标
y = rho * np.sin(theta)  # 转换为直角坐标的纵坐标

# 使用最小二乘法拟合圆，优化的目标函数是点到圆的距离之和
# 初始值设为(0, 0, 1)，即圆心在原点，半径为1
# 使用scipy.optimize.minimize函数，方法选择L-BFGS-B，可以设置变量的范围
# 例如，圆心的极径应该大于0，半径也应该大于0
res = opt.minimize(lambda p: np.sum(dist_to_circle(p[:2], p[2], theta) ** 2), (0, 0, 1), method='L-BFGS-B',
                   bounds=((0, None), (0, 2 * np.pi), (0, None)))
c_fit = res.x[:2]  # 拟合的圆心的极坐标
r_fit = res.x[2]  # 拟合的半径
print('拟合的圆心的极坐标为：', c_fit)
print('拟合的半径为：', r_fit)

# 将拟合的圆心的极坐标转换为直角坐标
c_fit_xy = (c_fit[0] * np.cos(c_fit[1]), c_fit[0] * np.sin(c_fit[1]))
print('拟合的圆心的直角坐标为：', c_fit_xy)

# 绘制散点图和拟合的圆
plt.scatter(x, y, label='data')  # 绘制散点图
x_fit = np.linspace(min(x), max(x), 100)  # 生成拟合的圆的横坐标
y_fit = circle_equation(c_fit_xy, r_fit, x_fit)  # 计算拟合的圆的纵坐标
plt.plot(x_fit, y_fit, label='fit')  # 绘制拟合的圆
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
