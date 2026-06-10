# 基于CNN的AM/FM调制方式识别系统 - 论文项目

## 项目概述

这是一个完整的Jupyter Notebook论文项目，包含了AM/FM调制方式识别系统的理论分析、代码实现、实验结果和结论。

## 文件说明

```
├── AM_FM_Modulation_Recognition_Paper.ipynb  # 论文Notebook（主文件）
├── start.bat                                   # Web应用启动脚本
├── app.py                                     # Flask后端
├── templates/
│   └── index.html                             # Web界面
└── modulation_recognition.py                   # Python脚本版本
```

## 如何使用Jupyter Notebook

### 方法1：直接运行Notebook（推荐）

1. **安装Jupyter Notebook**：
```bash
pip install jupyter notebook
```

2. **启动Jupyter**：
```bash
jupyter notebook
```

3. **打开Notebook**：
- 浏览器会自动打开
- 或者访问 http://localhost:8888
- 选择 `AM_FM_Modulation_Recognition_Paper.ipynb` 文件

### 方法2：使用JupyterLab

```bash
pip install jupyterlab
jupyter lab
```

### 方法3：导出为PDF/HTML

```bash
# 导出为HTML
jupyter nbconvert --to html AM_FM_Modulation_Recognition_Paper.ipynb

# 导出为PDF
jupyter nbconvert --to pdf AM_FM_Modulation_Recognition_Paper.ipynb
```

## Notebook内容结构

### 第1部分：引言
- 研究背景
- 研究目标

### 第2部分：理论基础
- AM调制原理
- FM调制原理
- 解调原理

### 第3部分：系统设计
- 系统架构
- 特征提取方法

### 第4部分：信号生成与调制
- 基本信号参数设置
- AM/FM调制实现
- 信号可视化

### 第5部分：信道噪声
- 高斯白噪声模型
- 不同SNR下的信号演示

### 第6部分：信号解调
- AM包络检波
- FM鉴频

### 第7部分：特征提取
- 6维统计特征
- 特征可视化

### 第8部分：模型训练
- 数据集生成
- SVM分类器训练
- 混淆矩阵

### 第9部分：性能分析
- 不同SNR下的准确率
- 性能曲线

### 第10部分：实时演示
- 随机信号识别

### 第11部分：结论
- 主要成果
- 实验结论
- 未来工作

### 第12部分：参考文献

## 依赖安装

Notebook运行所需的Python包：

```bash
pip install numpy matplotlib scipy scikit-learn jupyter notebook seaborn
```

或者一次性安装：

```bash
pip install numpy matplotlib scipy scikit-learn jupyter notebook seaborn flask
```

## 注意事项

1. **首次运行**：Notebook会自动安装所有依赖
2. **代码单元格**：按 `Shift + Enter` 运行当前单元格
3. **重启内核**：如果遇到问题，选择 Kernel → Restart & Run All
4. **保存**：定期按 `Ctrl + S` 保存Notebook

## Web应用使用

如果想要使用Web界面版本的系统：

1. 双击 `start.bat`
2. 浏览器会自动打开 http://127.0.0.1:5000
3. 可以交互式地生成信号和测试AI识别

## 作者信息

- **Author**: mimi
- **Date**: 2026-05-28

## 联系方式

- Email: 3322346701@qq.com
- GitHub: https://github.com/MiMimimimi777/2D-CNN-based-AM-FM-Modulation-Signal-Recognition #还没做