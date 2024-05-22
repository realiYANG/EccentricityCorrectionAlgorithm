# 导入必要的库
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
import las


# 定义函数
def dist_to_circle_correct(c, r, x, y):
	"""计算点 (x, y) 到圆心 c 半径为 r 的圆的距离的函数"""
	return np.sqrt((x - c[0]) ** 2 + (y - c[1]) ** 2) - r


def generate_circle_points(c, r, n):
	"""生成圆周上 n 个点的坐标的函数"""
	theta = np.linspace(0, 2 * np.pi, n)  # 生成 0 到 2π 之间均匀分布的 n 个角度
	x = c[0] + r * np.cos(theta)  # 计算圆周上点的 x 坐标
	y = c[1] + r * np.sin(theta)  # 计算圆周上点的 y 坐标
	return x, y


log = las.LASReader('ning209H13-4_resampled_3200-4000.las', null_subs=np.nan)

# 设置真实圆参数
c_true = (0, 0)  # 真实圆心坐标
r_true = 10  # 真实圆半径
n = 24  # 生成的点数
theta = np.linspace(0, 2 * np.pi, n, endpoint=False)  # 生成 0 到 2π 之间均匀分布的 n 个角度
rho = r_true + np.random.normal(0, 0.5, n)  # 带噪声的半径值
x = rho * np.cos(theta) - 2  # 计算带噪声的点的 x 坐标
y = rho * np.sin(theta) - 3  # 计算带噪声的点的 y 坐标

# 拟合圆
res = opt.minimize(lambda p: np.sum(dist_to_circle_correct(p[:2], p[2], x, y) ** 2), (0, 0, 1), method='L-BFGS-B',
                   bounds=None)
"""使用 L-BFGS-B 优化算法拟合圆，最小化点到圆的距离的平方和"""
c_fit = res.x[:2]  # 拟合出的圆心坐标
r_fit = res.x[2]  # 拟合出的圆半径
print('拟合的圆心的直角坐标为：', c_fit)
print('拟合的半径为：', r_fit)

# 生成拟合圆点
x_fit, y_fit = generate_circle_points(c_fit, r_fit, 100)  # 生成拟合圆上 100 个点的坐标

# 绘制结果
plt.scatter(x, y, label='data')  # 散点图显示原始带噪声的点
plt.plot(x_fit, y_fit, label='fit')  # 绘制拟合圆
plt.xlabel('x')
plt.ylabel('y')
plt.legend()  # 显示图例
plt.xlim(-15, 15)
plt.ylim(-15, 15)
# 显示 x 和 y 轴的线
plt.axhline(0, color='black', linewidth=0.5)  # 显示 x 轴的线，颜色为黑色，线宽为 0.5
plt.axvline(0, color='black', linewidth=0.5)  # 显示 y 轴的线，颜色为黑色，线宽为 0.5
# 设置 x 和 y 轴的数据间隔为 1
plt.xticks(np.arange(-15, 16, 2))
plt.yticks(np.arange(-15, 16, 2))

plt.scatter(c_fit[0], c_fit[1], color='red', marker='o')  # 在圆心位置绘制一个红色圆点
# 添加注释
plt.annotate('Center' + '(' + str(round(c_fit[0], 2)) + ',' + str(round(c_fit[1], 2)) + ')', (c_fit[0], c_fit[1]),
             xytext=(c_fit[0] + 1, c_fit[1] + 1),
             arrowprops=dict(facecolor='black', arrowstyle='->'), fontsize=12)

plt.show()  # 显示图像
