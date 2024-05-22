# 导入必要的库
import numpy as np
import las

log = las.LASReader('ning209H13-4_resampled_3200-4000.las', null_subs=np.nan)
print(log)