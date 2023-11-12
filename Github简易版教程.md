# Github项目

## 1. 项目克隆的clone操作

1). 在本地/服务器端新建“项目文件”（我命名为AI_Project，大家也可以命名为其他的名字），cd到“项目文件”路径下，然后克隆仓库：

```bash
cd ${AI_PROJECT}
git clone https://github.com/Iris1946/AI_2023Fall.git
```

2). 在“项目文件”下创建“数据集文件”（我命名为AI_Data，大家也可以命名为其他的名字），cd到“数据集文件”路径下，然后从https://www.datafountain.cn/competitions/530/datasets下载训练集和测试集：

3). 项目的目录结构如下：

├── ${AI_PROJECT}
│   ├── AI_2023Fall
│       ├── baseline
│       └── README.md
│   ├── ${AI_DATA}
│       ├── test_dataset
│       └── train_dataset
│           ├── train_internet.csv
│           └── train_public.csv

## 2. 修改代码后的push操作

1). cd到AI_2023Fall文件下，新建以自己名字命名的branch_name（如yuqing）

```bash
cd ${AI_PROJECT}/AI_2023Fall
git checkout -b ${BRANCH_NAME}
```

2). 使用`git status`查看目前所在的分支，注意不要在主分支`main`上

```bash
git status

# On branch yuqing
# Untracked files:
#   (use "git add <file>..." to include in what will be committed)

#         hello.py

# nothing added to commit but untracked files present (use "git add" to track)
```

3). 使用`push`操作将改动推到远程仓库上

```bash
git add .
git commit -m "${COMMIT CONTENT}"
git push origin ${BRANCH_NAME}
```

注意，`push`的时候可以出现`conflict`，这时需要大家解除冲突后再重复`add`, `commit`, `push`操作

4). 在github仓库中找到`Pull Request` > `New Pull Request` 。界面中的`base`选择主分支`main`，`compare`选择以自己名字命名的分支`${BRANCH_NAME}`，即`base:main`->`compare:${BRANCH_NAME}`。最后点击`Create pull request`。

## 3. 每次更改前的pull操作

为避免`push`时需要花费时间解冲突，请大家每次修改前务必`pull`一下主分支～

```bash
git pull origin main
```