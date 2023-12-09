import warnings
import pandas as pd
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

train = pd.read_csv('/ai/leiyuqing/workspace/Course/AI_Project/data/clean_data/train.csv')
test = pd.read_csv('/ai/leiyuqing/workspace/Course/AI_Project/data/clean_data/test.csv')

# 1. 数据分割

X_train_val = train.drop(['is_default','loan_id'], axis = 1, inplace = False)
y_train_val = train['is_default']

X_test = test.drop(['loan_id'], axis = 1, inplace = False)

X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.3, random_state=42)

# 2. 归一化

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# 3. 集成学习

# 创建一个字典用于存储模型结果
results_dict = {'Model': [], 'AUC': [], 'Accuracy': []}

# 计算AUC和准确率的函数，并更新字典
def evaluate_model(model_name, auc, accuracy):    
    results_dict['Model'].append(model_name)
    results_dict['AUC'].append(auc)
    results_dict['Accuracy'].append(accuracy)

print("开始训练...")

## 3.1 梯度提升机
print("=" * 100)
print("正在训练梯度提升机...")
from sklearn.ensemble import GradientBoostingClassifier

# 创建梯度提升分类器
gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)

# 训练模型
gb_model.fit(X_train, y_train)

# 预测验证集
y_val_pred = gb_model.predict(X_val)

# 计算准确率
accuracy = 0
accuracy += sum(y == y_pred for y, y_pred in zip(y_val, y_val_pred))
accuracy /= len(y_val)

print(f"梯度提升机准确率: {accuracy * 100:.2f}%")

# 计算AUC
auc = roc_auc_score(y_val, y_val_pred)
print(f"梯度提升机AUC: {auc * 100:.2f}%")
evaluate_model('GradientBoosting', auc*100, accuracy*100)

# 计算ROC曲线
fpr, tpr, thresholds = roc_curve(y_val, y_val_pred)

# 绘制ROC曲线
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('GradientBoosting ROC Curve')
plt.legend()
plt.savefig('/ai/leiyuqing/workspace/Course/AI_Project/data/submit/roc/1_GradientBoost.png', dpi=300)

## 3.2 LightBGM
print("=" * 100)
print("正在训练LightBGM...")

import lightgbm as lgb

# 创建LightGBM分类器，并使用GPU加速
lightgbm_model = lgb.LGBMClassifier(n_estimators=200)

# 训练模型
lightgbm_model.fit(X_train, y_train, eval_set=[(X_val, y_val)])

# 预测验证集
y_val_pred = lightgbm_model.predict(X_val, num_iteration=lightgbm_model.best_iteration_)

# 计算准确率
accuracy = 0
accuracy += sum(y == y_pred for y, y_pred in zip(y_val, y_val_pred))
accuracy /= len(y_val)

print(f"LightBGM准确率: {accuracy * 100:.2f}%")

# 计算AUC
auc = roc_auc_score(y_val, y_val_pred)
print(f"LightBGM AUC: {auc * 100:.2f}%")
evaluate_model('LightBGM', auc*100, accuracy*100)

# 计算ROC曲线
fpr, tpr, thresholds = roc_curve(y_val, y_val_pred)

# 绘制ROC曲线
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('LightBGM ROC Curve')
plt.legend()
plt.savefig('/ai/leiyuqing/workspace/Course/AI_Project/data/submit/roc/2_LightBGM.png', dpi=300)

## 3.3 CatBoost
print("=" * 100)
print("正在训练CatBoost...")

from catboost import CatBoostClassifier, Pool

# 创建CatBoost分类器，并使用GPU加速
catboost_model = CatBoostClassifier(iterations=100, depth=6, learning_rate=0.1, loss_function='Logloss', task_type='GPU', verbose=10)

# 将数据包装成CatBoost的数据池
train_pool = Pool(data=X_train, label=y_train)
val_pool = Pool(data=X_val, label=y_val)

# 训练模型
catboost_model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=10)

# 预测验证集
y_val_pred = catboost_model.predict(X_val)

# 计算准确率
accuracy = 0
accuracy += sum(y == y_pred for y, y_pred in zip(y_val, y_val_pred))
accuracy /= len(y_val)

print(f"CatBoost准确率: {accuracy * 100:.2f}%")

# 计算AUC
auc = roc_auc_score(y_val, y_val_pred)
print(f"CatBoost AUC: {auc * 100:.2f}%")
evaluate_model('CatBoost', auc*100, accuracy*100)

# 计算ROC曲线
fpr, tpr, thresholds = roc_curve(y_val, y_val_pred)

# 绘制ROC曲线
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('CatBoost ROC Curve')
plt.legend()
plt.savefig('/ai/leiyuqing/workspace/Course/AI_Project/data/submit/roc/3_CatBoost.png', dpi=300)

## 3.4 XGBoost
print("=" * 100)
print("正在训练XGBoost...")
import xgboost as xgb

# 定义XGBoost分类器的参数
params = {
    'objective': 'binary:logistic',
    'max_depth': 6,
    'learning_rate': 0.1,
    'eval_metric': 'logloss'
}

# 转换数据为XGBoost的DMatrix格式
dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)

# 训练XGBoost模型
num_rounds = 100
xgb_model = xgb.train(params, dtrain, num_rounds, evals=[(dval, 'eval')], early_stopping_rounds=10, verbose_eval=True)

# 预测验证集
y_pred_probs = xgb_model.predict(dval)
y_val_pred = [1 if prob > 0.5 else 0 for prob in y_pred_probs]

# 计算准确率
accuracy = 0
accuracy += sum(y == y_pred for y, y_pred in zip(y_val, y_val_pred))
accuracy /= len(y_val)

print(f"XGBoost准确率: {accuracy * 100:.2f}%")

# 计算AUC
auc = roc_auc_score(y_val, y_val_pred)
print(f"XGBoost AUC: {auc * 100:.2f}%")
evaluate_model('XGBoost', auc*100, accuracy*100)


# 计算ROC曲线
fpr, tpr, thresholds = roc_curve(y_val, y_val_pred)

# 绘制ROC曲线
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('XGBoost ROC Curve')
plt.legend()
plt.savefig('/ai/leiyuqing/workspace/Course/AI_Project/data/submit/roc/4_XGBoost.png', dpi=300)

## 3.5 随机森林
print("=" * 100)
print("正在训练随机森林...")

from sklearn.ensemble import RandomForestClassifier

# 定义随机森林分类器的参数
rf_params = {
    'n_estimators': 100,  # 树的数量
    'max_depth': 6,       # 每棵树的最大深度
    'random_state': 42    # 随机种子，用于复现结果
}

# 创建随机森林分类器
rf_model = RandomForestClassifier(**rf_params)

# 训练模型
rf_model.fit(X_train, y_train)

# 预测验证集
y_val_pred = rf_model.predict(X_val)

# 计算准确率
accuracy = 0
accuracy += sum(y == y_pred for y, y_pred in zip(y_val, y_val_pred))
accuracy /= len(y_val)

print(f"随机森林准确率: {accuracy * 100:.2f}%")

# 计算AUC
auc = roc_auc_score(y_val, y_val_pred)
print(f"随机森林AUC: {auc * 100:.2f}%")
evaluate_model('随机森林', auc*100, accuracy*100)

# 计算ROC曲线
fpr, tpr, thresholds = roc_curve(y_val, y_val_pred)

# 绘制ROC曲线
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Random Forest ROC Curve')
plt.legend()
plt.savefig('/ai/leiyuqing/workspace/Course/AI_Project/data/submit/roc/5_RandomForest.png', dpi=300)

## 3.6 AdaBoost
print("=" * 100)
print("正在训练AdaBoost...")

from sklearn.ensemble import AdaBoostClassifier

# 定义AdaBoost分类器的参数
adaboost_params = {
    'n_estimators': 50,    # 弱分类器的数量
    'learning_rate': 0.1,  # 学习率
    'random_state': 42     # 随机种子，用于复现结果
}

# 创建AdaBoost分类器
adaboost_model = AdaBoostClassifier(**adaboost_params)

# 训练模型
adaboost_model.fit(X_train, y_train)

# 预测验证集
y_val_pred = adaboost_model.predict(X_val)

# 计算准确率
accuracy = 0
accuracy += sum(y == y_pred for y, y_pred in zip(y_val, y_val_pred))
accuracy /= len(y_val)

print(f"AdaBoost准确率: {accuracy * 100:.2f}%")

# 计算AUC
auc = roc_auc_score(y_val, y_val_pred)
print(f"AdaBoost AUC: {auc * 100:.2f}%")
evaluate_model('AdaBoost', auc*100, accuracy*100)

# 计算ROC曲线
fpr, tpr, thresholds = roc_curve(y_val, y_val_pred)

# 绘制ROC曲线
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('AdaBoost ROC Curve')
plt.legend()
plt.savefig('/ai/leiyuqing/workspace/Course/AI_Project/data/submit/roc/6_AdaBoost.png', dpi=300)

## 3.7 Stacking

# from xgboost import XGBClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import StackingClassifier

# # 定义基础模型
# base_models = [
#     ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
#     ('xgb', XGBClassifier(tree_method='gpu_hist', random_state=42))
# ]

# # 定义次级模型
# meta_model = LogisticRegression()

# # 创建Stacking分类器
# stacking_model = StackingClassifier(estimators=base_models, final_estimator=meta_model)

# # 训练Stacking模型
# stacking_model.fit(X_train, y_train)

# # 预测验证集
# y_val_pred = stacking_model.predict(X_val)

# # 计算准确率
# accuracy = 0
# accuracy += sum(y == y_pred for y, y_pred in zip(y_val, y_val_pred))
# accuracy /= len(y_val)

# print(f"Staking准确率: {accuracy * 100:.2f}%")

## 3.8 Voting

# from sklearn.ensemble import VotingClassifier

# # 定义基础模型
# base_models = [
#     ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
#     ('xgb', XGBClassifier(use_label_encoder=False, tree_method='gpu_hist', random_state=42)),
#     ('lgbm', LGBMClassifier(device='gpu', random_state=42))
# ]

# # 创建Voting分类器
# voting_model = VotingClassifier(estimators=base_models, voting='hard')  # 'hard'表示投票决策方式，也可以选择 'soft'

# # 训练Voting模型
# voting_model.fit(X_train, y_train)

# # 预测验证集
# y_val_pred = voting_model.predict(X_val)

# # 计算准确率
# accuracy = 0
# accuracy += sum(y == y_pred for y, y_pred in zip(y_val, y_val_pred))
# accuracy /= len(y_val)

# print(f"Staking准确率: {accuracy * 100:.2f}%")

# 4. 二分类模型

## 4.1 神经网络
print("=" * 100)
print("正在训练神经网络...")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 转换为PyTorch的Tensor并移动到GPU
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).to(device)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).to(device)

# 创建数据集和数据加载器
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 定义神经网络模型并移动到GPU
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(X_train.shape[1], 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x

model = SimpleNN().to(device)

# 创建模型实例、损失函数和优化器
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels.view(-1, 1))
        loss.backward()
        optimizer.step()

# 评估模型
model.eval()
val_outputs = model(X_val_tensor)
y_pred_probs = val_outputs.cpu().detach().numpy()
y_val_pred = [1 if prob > 0.5 else 0 for prob in y_pred_probs]

# 计算准确率
accuracy = 0
accuracy = sum(y == y_pred for y, y_pred in zip(y_val, y_val_pred))
accuracy = accuracy / len(y_val)
print(f"神经网络准确率: {accuracy * 100:.2f}%")

# 计算AUC
auc = roc_auc_score(y_val, y_val_pred)
print(f"神经网络 AUC: {auc * 100:.2f}%")
evaluate_model('神经网络', auc*100, accuracy*100)

# 计算ROC曲线
fpr, tpr, thresholds = roc_curve(y_val, y_val_pred)

# 绘制ROC曲线
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Neural Network ROC Curve')
plt.legend()
plt.savefig('/ai/leiyuqing/workspace/Course/AI_Project/data/submit/roc/6_NN.png', dpi=300)

## 4.2 Logistic回归
print("=" * 100)
print("正在训练逻辑回归...")
from sklearn.linear_model import LogisticRegression

logistic_regression_model = LogisticRegression(random_state=42)
logistic_regression_model.fit(X_train, y_train)
y_val_pred = logistic_regression_model.predict(X_val)

# 计算准确率
accuracy = 0
accuracy += sum(y == y_pred for y, y_pred in zip(y_val, y_val_pred))
accuracy /= len(y_val)

print(f"逻辑回归准确率: {accuracy * 100:.2f}%")

# 计算AUC
auc = roc_auc_score(y_val, y_val_pred)
print(f"逻辑回归AUC: {auc * 100:.2f}%")
evaluate_model('逻辑回归', auc*100, accuracy*100)

# 计算ROC曲线
fpr, tpr, thresholds = roc_curve(y_val, y_val_pred)

# 绘制ROC曲线
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Logistic Regression ROC Curve')
plt.legend()
plt.savefig('/ai/leiyuqing/workspace/Course/AI_Project/data/submit/roc/7_LogisticRegression.png', dpi=300)

## 4.3 决策树
print("=" * 100)
print("正在训练决策树...")
from sklearn.tree import DecisionTreeClassifier

decision_tree_model = DecisionTreeClassifier(random_state=42)
decision_tree_model.fit(X_train, y_train)
y_val_pred = decision_tree_model.predict(X_val)

# 计算准确率
accuracy = 0
accuracy += sum(y == y_pred for y, y_pred in zip(y_val, y_val_pred))
accuracy /= len(y_val)

print(f"决策树准确率: {accuracy * 100:.2f}%")

# 计算AUC
auc = roc_auc_score(y_val, y_val_pred)
print(f"决策树AUC: {auc * 100:.2f}%")
evaluate_model('决策树', auc*100, accuracy*100)

# 计算ROC曲线
fpr, tpr, thresholds = roc_curve(y_val, y_val_pred)

# 绘制ROC曲线
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Decision Tree ROC Curve')
plt.legend()
plt.savefig('/ai/leiyuqing/workspace/Course/AI_Project/data/submit/roc/8_DecisionTree.png', dpi=300)

## 4.5 K近邻算法
print("=" * 100)
print("正在训练KNN...")
from sklearn.neighbors import KNeighborsClassifier

knn_model = KNeighborsClassifier()
knn_model.fit(X_train, y_train)
y_val_pred = knn_model.predict(X_val)

accuracy = 0
accuracy += sum(y == y_pred for y, y_pred in zip(y_val, y_val_pred))
accuracy = float(accuracy / len(y_val))
print(f"KNN准确率: {accuracy * 100:.2f}%")

# 计算AUC
auc = roc_auc_score(y_val, y_val_pred)
print(f"KNN AUC: {auc * 100:.2f}%")
evaluate_model('KNN', auc*100, accuracy*100)

# 计算ROC曲线
fpr, tpr, thresholds = roc_curve(y_val, y_val_pred)

# 绘制ROC曲线
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('KNN ROC Curve')
plt.legend()
plt.savefig('/ai/leiyuqing/workspace/Course/AI_Project/data/submit/roc/9_KNN.png', dpi=300)

## 4.6 朴素贝叶斯
print("=" * 100)
print("正在训练朴素贝叶斯...")

from sklearn.naive_bayes import GaussianNB

naive_bayes_model = GaussianNB()
naive_bayes_model.fit(X_train, y_train)
y_val_pred = naive_bayes_model.predict(X_val)

# 计算准确率
accuracy = 0
accuracy += sum(y == y_pred for y, y_pred in zip(y_val, y_val_pred))
accuracy /= len(y_val)

print(f"朴素贝叶斯准确率: {accuracy * 100:.2f}%")

# 计算AUC
auc = roc_auc_score(y_val, y_val_pred)
print(f"朴素贝叶斯AUC: {auc * 100:.2f}%")
evaluate_model('朴素贝叶斯', auc*100, accuracy*100)

# 计算ROC曲线
fpr, tpr, thresholds = roc_curve(y_val, y_val_pred)

# 绘制ROC曲线
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Bayes ROC Curve')
plt.legend()
plt.savefig('/ai/leiyuqing/workspace/Course/AI_Project/data/submit/roc/10_Bayes.png', dpi=300)


## 4.4 SVM
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

print("=" * 100)
print("正在训练SVM...")

# 转换为PyTorch的张量并移动到GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 转换为PyTorch的Tensor并移动到GPU
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.int64).to(device)  # 将标签的数据类型改为torch.int64
X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
y_val_tensor = torch.tensor(y_val.values, dtype=torch.int64).to(device)  # 将标签的数据类型改为torch.int64


# 创建数据集和数据加载器
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 定义线性SVM模型并移动到GPU
class LinearSVM(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LinearSVM, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x):
        return self.linear(x)

# 初始化模型、损失函数和优化器，并移动到GPU
input_size = X_train.shape[1]
num_classes = 2
svm_model = LinearSVM(input_size, num_classes).to(device)
criterion = nn.MultiMarginLoss()
optimizer = optim.SGD(svm_model.parameters(), lr=0.01)

# 训练模型
num_epochs = 10
for epoch in range(num_epochs):
    svm_model.train()
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = svm_model(inputs)
        
        # 将标签维度调整为一维
        labels = labels.view(-1)
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()


# 预测并评估模型
with torch.no_grad():
    y_val_pred = torch.argmax(svm_model(X_val_tensor), dim=1)
    y_val_pred = y_val_pred.cpu()

# 计算准确率
accuracy = 0
accuracy = sum(y == y_pred for y, y_pred in zip(y_val, y_val_pred))
accuracy = accuracy / len(y_val)
print(f"SVM准确率: {accuracy * 100:.2f}%")

# 计算AUC
auc = roc_auc_score(y_val, y_val_pred)
print(f"SVM AUC: {auc * 100:.2f}%")
evaluate_model('SVM', auc*100, accuracy.item()*100)

# 计算ROC曲线
fpr, tpr, thresholds = roc_curve(y_val, y_val_pred)

# 绘制ROC曲线
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('SVM ROC Curve')
plt.legend()
plt.savefig('/ai/leiyuqing/workspace/Course/AI_Project/data/submit/roc/11_SVM.png', dpi=300)

print("=" * 100)
print("结束训练，打印训练结果...")

## 模型结果
# 将结果字典转换为数据框
results_df = pd.DataFrame(results_dict)

# 将结果保存到CSV文件
results_df.to_csv('/ai/leiyuqing/workspace/Course/AI_Project/data/submit/results.csv', index=False)

# 显示结果数据框
print(results_df)


# 模型预测

## 使用XGBoost进行预测

# 转换测试集为XGBoost的DMatrix格式
dtest = xgb.DMatrix(X_test)

# 预测测试集
y_test_pred_probs = xgb_model.predict(dtest)
y_test_pred = [1 if prob > 0.5 else 0 for prob in y_test_pred_probs]

# 输出预测结果
print("=" * 100)
print("使用XGBoost的预测结果:")

# 将测试集 loan_id 列作为第一列，并重命名为 'id'
result_df = pd.DataFrame({'id': test['loan_id'], 'is_default': y_test_pred})

# 将结果保存到 submit.csv 文件
result_df.to_csv('/ai/leiyuqing/workspace/Course/AI_Project/data/submit/submit.csv', index=False)

# 显示结果DataFrame的前几行
print(result_df.head())





