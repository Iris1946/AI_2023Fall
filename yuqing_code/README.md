# version 1 说明

## 1. `my_dataprocessing.ipynb`

### 运行环境：

需要在`google colab`（需科学上网）或`jupyter notebook`（需下载Anaconda）上运行，不需要gpu

PS：如果打不开ipynb文件的童鞋，可以打开同名py文件运行

### 数据准备：

参考`CCF2-个贷违约预测Baseline`：

- 先将` train_public.csv `另存为` train_public2.csv`，并对**earlies_credit_mon**改成**短日期**格式 ！！(`train_internet.csv`和`test.csv`同理)
- 将短日期格式的 2021/12/1 => 2001-12-01 （这里2021应该是系统自动添加上的，实际为 12/1，即月/年）

### 路径更改：

开头的数据读取路径需要更改：

```python
train_bank = pd.read_csv('drive/MyDrive/AI_Project/data/train_dataset/train_public2.csv')
train_internet = pd.read_csv('drive/MyDrive/AI_Project/data/train_dataset/train_internet2.csv')
test = pd.read_csv('drive/MyDrive/AI_Project/data/test_public2.csv')
```

结尾存储预处理完的数据时也需要更改路径：

```python
train_filtered.to_csv('/content/drive/MyDrive/AI_Project/data/clean_data/train.csv', index=False)
test_filtered.to_csv('/content/drive/MyDrive/AI_Project/data/clean_data/test.csv', index=False)
```

### 可以完善的地方：

1. 交叉验证
2. 数据扩充
3. 数据不平衡问题（惩罚系数、好人坏人、重采样等）
4. train_public和train_internet的数据分布不同，可以考虑把train_set拆开

## 2. `my_model.py`

### 运行环境：

pycharm等，需使用`GPU`加速；用到库及其版本号见`requirement.txt`

### 路径更改：

开头数据读取的路径需要更改：

```python
train = pd.read_csv('/ai/leiyuqing/workspace/Course/AI_Project/data/clean_data/train.csv')
test = pd.read_csv('/ai/leiyuqing/workspace/Course/AI_Project/data/clean_data/test.csv')
```

每一个模型评估时绘制的`roc`曲线存储路径均需修改（例如GardientBoost）：

```python
plt.savefig('/ai/leiyuqing/workspace/Course/AI_Project/data/submit/roc/1_GradientBoost.png', dpi=300)
```

结尾 存储各个模型的accuarcy和auc的`result`文件地址需要修改：

```python
results_df.to_csv('/ai/leiyuqing/workspace/Course/AI_Project/data/submit/results.csv', index=False)
```

最后 预测test set情况的`submit`文件地址需要修改：

```python
result_df.to_csv('/ai/leiyuqing/workspace/Course/AI_Project/data/submit/submit.csv', index=False)
```

### 运行日志（可供参考）：

`roc曲线`、`模型性能比较（result.csv）`和`预测结果（submit.csv）`均运行结果上传至`model_result`文件夹下

```
开始训练...
====================================================================================================
正在训练梯度提升机...
梯度提升机准确率: 81.55%
梯度提升机AUC: 58.51%
====================================================================================================
正在训练LightBGM...
[LightGBM] [Info] Number of positive: 106132, number of negative: 425868
[LightGBM] [Info] Auto-choosing row-wise multi-threading, the overhead of testing was 0.012081 seconds.
You can set `force_row_wise=true` to remove the overhead.
And if memory is not enough, you can set `force_col_wise=true`.
[LightGBM] [Info] Total Bins 2692
[LightGBM] [Info] Number of data points in the train set: 532000, number of used features: 20
[LightGBM] [Info] [binary:BoostFromScore]: pavg=0.199496 -> initscore=-1.389446
[LightGBM] [Info] Start training from score -1.389446
LightBGM准确率: 81.60%
LightBGM AUC: 59.78%
====================================================================================================
正在训练CatBoost...
0:      learn: 0.6218768        test: 0.6217794 best: 0.6217794 (0)     total: 6.04ms   remaining: 598ms
10:     learn: 0.4327535        test: 0.4318108 best: 0.4318108 (10)    total: 57.3ms   remaining: 464ms
20:     learn: 0.4126165        test: 0.4114150 best: 0.4114150 (20)    total: 109ms    remaining: 411ms
30:     learn: 0.4070929        test: 0.4058359 best: 0.4058359 (30)    total: 161ms    remaining: 359ms
40:     learn: 0.4050704        test: 0.4038686 best: 0.4038686 (40)    total: 214ms    remaining: 309ms
50:     learn: 0.4036140        test: 0.4024747 best: 0.4024747 (50)    total: 268ms    remaining: 258ms
60:     learn: 0.4028103        test: 0.4017769 best: 0.4017769 (60)    total: 324ms    remaining: 207ms
70:     learn: 0.4021516        test: 0.4012006 best: 0.4012006 (70)    total: 379ms    remaining: 155ms
80:     learn: 0.4016507        test: 0.4007916 best: 0.4007916 (80)    total: 436ms    remaining: 102ms
90:     learn: 0.4013032        test: 0.4005420 best: 0.4005420 (90)    total: 490ms    remaining: 48.5ms
99:     learn: 0.4009534        test: 0.4003211 best: 0.4003211 (99)    total: 538ms    remaining: 0us
bestTest = 0.4003211349
bestIteration = 99
CatBoost准确率: 81.59%
CatBoost AUC: 58.94%
====================================================================================================
正在训练XGBoost...
[0]     eval-logloss:0.48687
[1]     eval-logloss:0.47505
[2]     eval-logloss:0.46540
[3]     eval-logloss:0.45732
[4]     eval-logloss:0.45046
[5]     eval-logloss:0.44470
[6]     eval-logloss:0.43978
[7]     eval-logloss:0.43543
[8]     eval-logloss:0.43177
[9]     eval-logloss:0.42854
[10]    eval-logloss:0.42577
[11]    eval-logloss:0.42327
[12]    eval-logloss:0.42112
[13]    eval-logloss:0.41922
[14]    eval-logloss:0.41756
[15]    eval-logloss:0.41610
[16]    eval-logloss:0.41479
[17]    eval-logloss:0.41360
[18]    eval-logloss:0.41244
[19]    eval-logloss:0.41148
[20]    eval-logloss:0.41062
[21]    eval-logloss:0.40973
[22]    eval-logloss:0.40900
[23]    eval-logloss:0.40834
[24]    eval-logloss:0.40773
[25]    eval-logloss:0.40720
[26]    eval-logloss:0.40669
[27]    eval-logloss:0.40626
[28]    eval-logloss:0.40588
[29]    eval-logloss:0.40551
[30]    eval-logloss:0.40515
[31]    eval-logloss:0.40486
[32]    eval-logloss:0.40454
[33]    eval-logloss:0.40428
[34]    eval-logloss:0.40400
[35]    eval-logloss:0.40378
[36]    eval-logloss:0.40353
[37]    eval-logloss:0.40333
[38]    eval-logloss:0.40312
[39]    eval-logloss:0.40294
[40]    eval-logloss:0.40276
[41]    eval-logloss:0.40257
[42]    eval-logloss:0.40246
[43]    eval-logloss:0.40233
[44]    eval-logloss:0.40221
[45]    eval-logloss:0.40208
[46]    eval-logloss:0.40199
[47]    eval-logloss:0.40187
[48]    eval-logloss:0.40179
[49]    eval-logloss:0.40167
[50]    eval-logloss:0.40160
[51]    eval-logloss:0.40150
[52]    eval-logloss:0.40142
[53]    eval-logloss:0.40129
[54]    eval-logloss:0.40122
[55]    eval-logloss:0.40111
[56]    eval-logloss:0.40104
[57]    eval-logloss:0.40100
[58]    eval-logloss:0.40096
[59]    eval-logloss:0.40089
[60]    eval-logloss:0.40082
[61]    eval-logloss:0.40077
[62]    eval-logloss:0.40068
[63]    eval-logloss:0.40064
[64]    eval-logloss:0.40057
[65]    eval-logloss:0.40054
[66]    eval-logloss:0.40049
[67]    eval-logloss:0.40046
[68]    eval-logloss:0.40039
[69]    eval-logloss:0.40036
[70]    eval-logloss:0.40032
[71]    eval-logloss:0.40029
[72]    eval-logloss:0.40028
[73]    eval-logloss:0.40025
[74]    eval-logloss:0.40024
[75]    eval-logloss:0.40020
[76]    eval-logloss:0.40018
[77]    eval-logloss:0.40015
[78]    eval-logloss:0.40009
[79]    eval-logloss:0.40007
[80]    eval-logloss:0.40004
[81]    eval-logloss:0.39999
[82]    eval-logloss:0.39997
[83]    eval-logloss:0.39991
[84]    eval-logloss:0.39989
[85]    eval-logloss:0.39985
[86]    eval-logloss:0.39985
[87]    eval-logloss:0.39984
[88]    eval-logloss:0.39979
[89]    eval-logloss:0.39976
[90]    eval-logloss:0.39975
[91]    eval-logloss:0.39972
[92]    eval-logloss:0.39970
[93]    eval-logloss:0.39969
[94]    eval-logloss:0.39969
[95]    eval-logloss:0.39967
[96]    eval-logloss:0.39966
[97]    eval-logloss:0.39962
[98]    eval-logloss:0.39961
[99]    eval-logloss:0.39961
XGBoost准确率: 81.61%
XGBoost AUC: 59.42%
====================================================================================================
正在训练随机森林...
随机森林准确率: 80.93%
随机森林AUC: 53.72%
====================================================================================================
正在训练AdaBoost...
AdaBoost准确率: 80.16%
AdaBoost AUC: 50.00%
====================================================================================================
正在训练神经网络...
神经网络准确率: 81.57%
神经网络 AUC: 59.90%
====================================================================================================
正在训练逻辑回归...
逻辑回归准确率: 80.71%
逻辑回归AUC: 55.41%
====================================================================================================
正在训练决策树...
决策树准确率: 73.24%
决策树AUC: 59.43%
====================================================================================================
正在训练KNN...
KNN准确率: 78.47%
KNN AUC: 58.55%
====================================================================================================
正在训练朴素贝叶斯...
朴素贝叶斯准确率: 31.10%
朴素贝叶斯AUC: 56.63%
====================================================================================================
正在训练SVM...
SVM准确率: 80.16%
SVM AUC: 50.00%
====================================================================================================
结束训练，打印训练结果...
               Model        AUC   Accuracy
0   GradientBoosting  58.514872  81.549123
1           LightBGM  59.782995  81.603947
2           CatBoost  58.944997  81.586404
3            XGBoost  59.421795  81.614474
4               随机森林  53.715671  80.932895
5           AdaBoost  50.000000  80.164912
6               神经网络  59.904246  81.566228
7               逻辑回归  55.409433  80.709649
8                决策树  59.430111  73.239474
9                KNN  58.553416  78.469298
10             朴素贝叶斯  56.632277  31.096930
11               SVM  50.000000  80.164909
====================================================================================================
使用XGBoost的预测结果:
        id  is_default
0  1000575           0
1  1028125           0
2  1010694           0
3  1026712           0
4  1002895           0
```

