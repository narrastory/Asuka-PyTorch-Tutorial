# 明日香 - Pytorch 快速入门保姆级教程(五)

`2026.03 | ming`

------

<div align="center">
  <img align="center" src="./media/index_5.jpg" alt="AI Generated Art" width="90%" style="border-radius: 10px; box-shadow: 0 4px 12px rgba(0,0,0,0.1);"/>
</div>


## 十一. 常用损失函数层

### 11.1 BCEWithLogitsLoss

在**二分类**问题中（比如判断一封邮件是否为垃圾邮件、一张图片中是否包含猫），我们通常希望模型输出一个概率值，表示样本属于正类的可能性。最常用的做法是在模型最后加上一个Sigmoid层，将输出压缩到0到1之间，然后使用二分类交叉熵作为损失函数。然而，直接串联Sigmoid和交叉熵在数值计算上可能存在不稳定性。在实际中我们也不会这么组合使用。

PyTorch 为此提供了一个优雅且高效的解决方案：**`nn.BCEWithLogitsLoss`**。它将 Sigmoid 层和二元交叉熵损失合并为一个操作，内部采用对数求和技巧保证数值稳定。因此，在处理二分类问题时，请务必优先使用 `nn.BCEWithLogitsLoss`，而不是手动组合 `Sigmoid + 交叉熵`。

下面我们用 `BCEWithLogitsLoss` 来完成一个简单的二分类任务。假设我们有一个小型数据集，包含 5 个样本，每个样本有 3 个特征。我们构建一个简单的三层全连接网络，输出一个 logit，然后计算损失。

```python
# 样本数据 (5个样本, 3个特征)
input_data = torch.tensor([
    [100,  0.3, 6.1],
    [121,  0.1, 1.8],
    [146, -0.4, 7.3],
    [239,  0.2, 5.5],
    [255,  0.1, 7.5]
], dtype=torch.float32)

# 二分类标签 (0 或 1)
target_data = torch.tensor([
    [0],
    [1],
    [0],
    [0],
    [1]
], dtype=torch.float32)

# 定义一个简单的全连接网络
model = nn.Sequential(
    nn.Linear(3, 5),
    nn.ReLU(),
    nn.Linear(5, 3),
    nn.ReLU(),
    nn.Linear(3, 1)   # 输出单个 logit
)

# 前向传播，得到 logits
logits = model(input_data)
print(logits)
# 输出示例:
# tensor([[-3.5913],
#         [-4.0318],
#         [-5.2216],
#         [-8.4370],
#         [-9.1208]])

# 创建损失函数
criterion = nn.BCEWithLogitsLoss()

# 计算损失
loss = criterion(logits, target_data)
print(loss)   # tensor(2.6406)
```

对于单个样本的单个类别，`BCEWithLogitsLoss` 计算的损失为：
$$
\ell(x,y) = -[w_p\cdot y \cdot \text{log} \space \sigma (x) + (1-y) \cdot \text{log}(1-\sigma(x))  ]
$$
其中：

- $x$ 是模型输出的 logit（未经过 Sigmoid 的原始值）。
- $y$ 是真实标签，取值为 0 或 1（也可以是 0~1 之间的任意数，用于标签平滑）。
- $\sigma(x) = \frac{1}{1 + e^{-x}}$ 是 Sigmoid 函数。
- $w_p$ 是正样本的权重（即 `pos_weight` 参数），若不指定则默认为 1。

但在实际实现中，PyTorch 并未直接计算 $\log \sigma(x)$ 和 $\log(1 - \sigma(x))$，而是采用数值稳定的等价形式，避免了因 $x$ 过大或过小导致的溢出，使计算过程更加稳定性。

**BCEWithLogitsLoss参数设置如下：**

| 参数         | 类型                | 说明                                                         |
| :----------- | :------------------ | :----------------------------------------------------------- |
| `pos_weight` | `Tensor` 或 `float` | 正样本的权重系数，用于提高正样本的权重。可以是一个标量（对所有类别相同），也可以是一个与类别数相同长度的张量（多标签分类时每个类别独立设置）。 |

多标签分类任务中，一个样本可以同时属于多个类别（例如一篇新闻文章可能同时属于“体育”和“科技”两个类别）。此时，模型的输出通常是一个形状为 `(N, C)` 的张量，其中 `C` 是类别总数。每个类别独立进行二分类，因此 `BCEWithLogitsLoss` 同样适用，只需保证 `target` 也是 `(N, C)` 的 0/1 矩阵即可。

下面是一个多标签分类的示例，假设有 3 个类别，批量大小为 4：

```python
# 模拟模型最后一层输出的 logits (batch_size=4, num_classes=3)
logits = torch.tensor([
    [1.5, -0.5, 2.1],
    [-0.8, 1.2, 0.3],
    [0.2, -1.1, 0.7],
    [1.1, 0.9, -0.4]
])

# 多标签真实标签 (0/1 矩阵)
target = torch.tensor([
    [1, 0, 1],   # 样本0属于类别0和2
    [0, 1, 0],   # 样本1属于类别1
    [0, 0, 1],   # 样本2属于类别2
    [1, 1, 0]    # 样本3属于类别0和1
], dtype=torch.float32)

# 若各类别样本不平衡，可以为每个类别设置不同的 pos_weight
pos_weight = torch.tensor([1.2, 0.8, 1.5])  # 类别0、1、2的权重

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
loss = criterion(logits, target)
print(loss)  # 例如 tensor(0.6732)
```

我知道很多同学在初次学习的时候搞不懂**`pos_weight`**是什么参数，具体作用是什么，下面就来解释一下：

在二分类任务（比如判断邮件是否为垃圾邮件、图像中是否有猫）中，我们经常遇到一个头疼的问题：**正负样本数量严重不均**。比如，10000封邮件中只有100封是垃圾邮件（正样本），其余9900封是正常邮件（负样本）。如果直接训练，模型可能会“偷懒”，把所有邮件都预测为正常邮件，这样准确率也能达到99%，但它实际上根本不会识别垃圾邮件，失去了实用价值。

`pos_weight` 参数就是专门为此设计的。它允许我们**提高正样本的损失权重**，让模型觉得分错一个正样本的代价远高于分错一个负样本，从而迫使模型更加关注那些稀少但重要的正样本。

在标准的二分类交叉熵中，每个样本（无论是正还是负）对损失的贡献是“平等”的。当负样本远多于正样本时，总的损失主要由负样本主导，模型就会倾向于把未知样本都判为负类，以最小化整体损失。

为了使模型向正样本倾斜，我们需要**提高正样本的权重**——也就是 `pos_weight` 的作用。

一个常用的启发式是：**让正样本的权重等于负样本数除以正样本数**。例如，负样本有9900个，正样本有100个，那么 `pos_weight = 9900 / 100 = 99`。这样，总的损失中正负样本的贡献大致平衡，模型就不会偏向负类了。

对于多标签分类（一个样本可能同时属于多个类别），`BCEWithLogitsLoss` 的输出形状是 `(N, C)`，其中 C 是类别数。此时 `pos_weight` 应该是一个长度为 C 的张量，每个元素对应那个类别的正样本权重。例如，三个类别的正样本比例不同，可以分别设置：

```python
pos_weight = torch.tensor([1.5, 3.0, 1.0])  # 类别0、1、2的权重
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
```

这样每个类别独立地调整权重，非常灵活。

### 11.2 CrossEntropyLoss

在多分类任务中（比如手写数字识别、图像分类、情感极性判断），我们希望模型输出一个概率分布，表示样本属于各个类别的可能性。最经典的做法是在模型的最后加上一个 Softmax 层，将 logits（原始输出）转换为概率，然后使用交叉熵损失计算与真实标签的差异。然而，直接串联 Softmax 和交叉熵在实践中的数值计算上存在不稳定性。PyTorch 为此提供了 **`nn.CrossEntropyLoss`**，它将 LogSoftmax 和负对数似然损失（NLLLoss）合并为一个操作，内部采用了更好的数值稳定算法。

因此，以后在处理多分类问题时，请直接使用 `nn.CrossEntropyLoss`，而不要手动组合 `Softmax + 交叉熵损失`。

下面是一个简单的多分类示例，假设有 5 个样本，每个样本分到 3 个类别之一。

```python
# 假设模型最后一层输出的 logits (batch_size=5, num_classes=3)
logits = torch.tensor([
    [2.0, 1.0, 0.1],
    [0.5, 2.5, 0.3],
    [1.3, 1.5, 2.0],
    [0.2, 0.1, 3.0],
    [2.2, 1.8, 0.5]
])

# 真实类别索引 (每个样本属于哪个类，这里是3分类任务)
# 注意：目标是类别索引，而非 one-hot 向量。目标值应为整数索引，不要传入 one-hot 编码的张量。
target = torch.tensor([0, 1, 2, 2, 0])  # 形状 (5,)

# 创建损失函数
criterion = nn.CrossEntropyLoss()

# 计算损失
loss = criterion(logits, target)
print(loss)  # tensor(0.4214)
```

**CrossEntropyLoss常用参数如下：**

| 参数           | 类型            | 说明                                                         |
| :------------- | :-------------- | :----------------------------------------------------------- |
| `weight`       | `Tensor` (可选) | 各类别的权重，用于处理类别不平衡问题。样本量少的类别可设置更高权重。 |
| `ignore_index` | `int` (可选)    | 指定一个类别索引，该类的样本不参与损失计算（常用于忽略“背景”类或无效数据）。 |

这里的`weight`参数和上文的`pos_weight`参数本质都是一样的。在实际任务中，各类别样本数量往往差异巨大。例如，一个三分类任务中，类别 0 有 1000 个样本，类别 1 有 100 个，类别 2 有 300 个。如果直接训练，模型会偏向样本多的类别。我们可以通过 `weight` 参数为每个类别赋予不同权重，让模型更关注样本少的类别。

```python
# 假设各类别样本数量：类别0:1000, 类别1:100, 类别2:300
# 下面这个权重的意思就是给类别1最高的权重4.6667，让模型多关注类别1，其它同理。
weight = torch.tensor([0.4667, 4.6667, 1.5556])

# 创建带权重的损失函数
criterion_weighted = nn.CrossEntropyLoss(weight=weight)

# 计算损失
loss_weighted = criterion_weighted(logits, target)
```

常用的权重计算公式：`该类权重 = 总样本数 / (类别数 * 该类样本数)`

`ignore_index`参数用的比较少，了解即可。在某些任务中，我们可能想忽略某些类别的样本。如下代码所示：

```python
# 假设类别索引 2 表示“忽略”
criterion = nn.CrossEntropyLoss(ignore_index=2)

# 目标中某些样本为 2
target = torch.tensor([3, 0, 2, 3, 0])

# 计算损失时，索引为 2 的样本会被自动跳过
loss = criterion(logits, target)
print(loss)
```


### 11.3 L1 Loss

L1损失函数，也称为平均绝对误差，是机器学习中最直观、最基础的损失函数之一。它直接衡量模型预测值与真实值之间的**绝对差异**。

对于单个样本，若预测值为 $\hat{y}_i$，真实值为 $y_i$，则该样本的损失为 $|\hat{y}_i - y_i|$。对整个数据集，L1损失计算所有样本绝对误差的平均值：
$$
\text{loss} = \frac{1}{n}\sum_{i=1}^{n} |\hat{y}_i-y_i|
$$
让我们用一组简单的数据来演示 `nn.L1Loss` 的用法。

```python
# 预测值和真实值
pred = torch.tensor([2.0, 3.0, 5.0])
target = torch.tensor([1.0, 2.0, 3.0])

l1_loss = nn.L1Loss()
loss = l1_loss(pred, target)
print(f"L1 Loss: {loss:.4f}")  # 输出: 1.3333
# 计算过程：(|2-1| + |3-2| + |5-3|) / 3 = (1 + 1 + 2) / 3 = 1.3333
```

**什么时候用L1损失？**

L1损失最大的优势在于**对离群点不敏感**。因为它对误差进行线性惩罚，不会像平方误差那样将大误差放大。如果数据集中存在少量异常值（如传感器故障产生的坏点），使用L1损失可以让模型更关注大多数正常样本，避免被个别离群点过度影响。

不过，L1损失也有一个理论上的小缺点：在误差为0处，它的导数不连续（左导数为-1，右导数为+1）。这意味着在极值点附近，梯度下降可能会产生轻微抖动，但在实际训练中通常影响不大，深度学习优化器（如SGD with momentum、Adam）能够很好地处理这种不连续性。

### 11.4 MSE Loss

均方误差是回归问题中最常用的损失函数之一。它将误差平方后再取平均，对大的误差施加更重的惩罚。
$$
\text{loss} = \frac{1}{n}\sum_{i=1}^{n} (\hat{y}_i-y_i)^2
$$
代码示例如下：

```python
pred = torch.tensor([2.0, 3.0, 5.0])
target = torch.tensor([1.0, 2.0, 3.0])

mse_loss = nn.MSELoss()
loss = mse_loss(pred, target)
print(f"MSE Loss: {loss:.4f}")  # 输出: 2.0000
# 计算过程：((2-1)^2 + (3-2)^2 + (5-3)^2) / 3 = (1 + 1 + 4) / 3 = 2.0
```

**MSE的特点**

1. **处处可导**：MSE在整个实数域上光滑可导，与误差成线性关系。这种光滑性使得梯度下降优化过程非常稳定。
2. **对离群点敏感**：由于平方运算，较大的误差会被进一步放大，从而在损失中占据主导地位。如果数据干净、离群点很少，这种特性可以帮助模型更快地减小大误差；但如果数据中混杂着较多异常值，模型可能会被这些离群点“带偏”，导致整体预测效果变差。

### 11.5 Smooth L1 Loss

在上一节中，我们学习了L1损失和MSE损失。两者各有优缺点：L1损失对离群点鲁棒，但在零点不可导；MSE损失处处可导，但对离群点过于敏感。那么有没有一种损失函数能“取长补短”，既具备L1的鲁棒性，又像MSE一样光滑可导呢？答案是肯定的，这就是本节要介绍的**平滑L1损失**。

平滑L1损失实际上是L1损失和L2损失的一个**折衷版本**。它通过一个阈值参数 $\beta$ 来控制损失函数的“形态”：

- 当预测值与真实值的**绝对误差小于 $\beta$** 时，采用平方误差（即MSE的形式），这样在零点附近是光滑的，便于梯度下降。
- 当绝对误差**大于等于 $\beta$** 时，采用线性误差（即L1的形式），这样对于大误差的惩罚不会像MSE那样被过分放大，从而对离群点更鲁棒。

你可以把平滑L1想象成一条“被修剪过的曲线”：在小误差区域它是抛物线（光滑），在大误差区域它变成了直线（稳健），两者在 $\beta$ 处平滑连接。

数学定义如下（$\beta$ 通常取1.0）：
$$
\mathrm{loss} = \left\{\begin{matrix}
0.5(\hat{y}_{i}-y_{i})^{2}/\beta, & \text{if} \space|\hat{y}_{i}-y_{i}|<\beta\\
|\hat{y}_{i}-y_{i}|-0.5\beta, & \text{otherwise}
\end{matrix}\right.
$$
示例代码如下：

```python
pred = torch.tensor([2.0, 3.0, 5.0])
target = torch.tensor([1.0, 2.0, 3.0])

smooth_l1 = nn.SmoothL1Loss(beta=1.0)  # beta默认值为1.0
loss = smooth_l1(pred, target)
print(f"Smooth L1 Loss (beta=1.0): {loss:.4f}")  # 输出: 0.8333
# 计算过程：
# 误差分别为：1.0, 1.0, 2.0
# 对于误差1.0 (< beta): 0.5 * 1.0^2 / 1.0 = 0.5
# 对于误差2.0 (>= beta): 2.0 - 0.5*1.0 = 1.5
# 平均损失：(0.5 + 0.5 + 1.5) / 3 = 2.5 / 3 ≈ 0.8333
```



## 十二. 自定义神经网络

### 12.1 自定义层

PyTorch 已经帮我们准备好了非常多的神经网络层，比如全连接层（`nn.Linear`）、卷积层（`nn.Conv2d`）、循环神经网络层（`nn.LSTM`）等等，足够我们搭建各种各样的常见网络。但有时候，你可能会遇到下面这些情况：

- 你想实现一篇最新论文中提出的新颖神经网络结构，但 PyTorch 还没有提供对应的层。
- 你自己发明了一种新的层类型，想要测试它的效果。
- 你想对现有的层做一些小改动，比如给某个激活函数增加一个可训练的参数。

这时候，就需要**自定义层**了。通过自定义层，你可以随心所欲地定义前向传播的逻辑，实现任意的神经网络结构。

> **注意**：除非你确实需要实现新颖的结构，否则建议优先使用 `torch.nn` 和 `torch.nn.functional` 中已有的层。它们经过了高度优化，稳定可靠，直接用它们能避免很多潜在的错误。

在 PyTorch 中，所有神经网络层、模型或者模型的某个部分，都是通过继承 `nn.Module` 类来构建的。你可以把 `nn.Module` 想象成一块乐高积木的**基座**——有了它，你可以拼出各种各样的零件（层），也可以把这些零件组合成一个更大的模型。

`nn.Module` 这个名字取得很巧妙，它叫“模块”而不是“层”或“模型”，因为它既可以是一个简单的层（比如 PyTorch 自带的 `nn.Linear`），也可以是一个复杂的模型（比如 AlexNet），还可以是模型中的一部分。这种灵活性让我们可以像搭积木一样自由地组合神经网络。

要自定义一个层，我们只需要继承 `nn.Module`，然后实现两个关键的方法：

- `__init__(self, ...)`：在这里定义层的参数（比如权重、偏置）以及包含的子模块。
- `forward(self, x)`：在这里编写前向传播的计算逻辑，也就是输入 `x` 如何经过这个层变成输出。

下面是一个最常见的模板：

```python
import torch
import torch.nn as nn

class CustomLayer(nn.Module):
    def __init__(self, *args):
        super().__init__()
        # 初始化参数和子模块
    
    def forward(self, x):
        # 定义前向传播逻辑
        return x
```

是不是很简单？接下来我们通过两种最常见的例子——**含参层**和**不含参层**——来深入理解。

**含参层的实现**

接下来我们就来手动实现一下PyTorch中自带的`nn.Linear()`层

```python
import torch
import torch.nn as nn

class MyLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features   # 输入特征数
        self.out_features = out_features # 输出特征数
        
        # 1. 定义权重参数：形状为 (out_features, in_features)
        #    nn.Parameter 会将普通的 Tensor 包装成可训练的参数，
        #    并自动添加到模块的参数列表中，方便优化器更新。
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        
        # 2. 定义偏置参数：形状为 (out_features)
        self.bias = nn.Parameter(torch.empty(out_features))
        
        # 3. 初始化参数
        #    合理的初始化能让训练更快、更稳定。这里用 Kaiming 均匀初始化，
        #    它是针对 ReLU 类激活函数设计的常用方法。
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        #    偏置通常初始化为 0
        nn.init.zeros_(self.bias)
    
    def forward(self, x):
        # 前向传播：x 的形状通常是 (batch_size, in_features)
        # 矩阵乘法：x @ weight^T ，然后加上偏置
        # 注意：weight 的形状是 (out_features, in_features)，转置后变成 (in_features, out_features)
        return x @ self.weight.t() + self.bias
    
```

**代码解释**

- **`nn.Parameter`**：这是 PyTorch 用来标记一个 `Tensor` 是**可训练参数**的方法。被 `nn.Parameter` 包装的 Tensor 会自动注册到模块的 `parameters()` 中，优化器就能找到它并更新它。总之只需要记住，凡是在自定义层中定义的可训练参数，都要用`nn.Parameter`包裹住。
- **初始化**：我们使用了 `nn.init.kaiming_uniform_` 对权重进行初始化。这里的下划线表示“原地操作”，也就是直接修改传入的 Tensor。Kaiming 初始化是一种常用的方法，它根据输入和输出的维度调整权重的范围，有助于防止梯度消失或爆炸。更多的权重初始化方法详见 `12.2` 小节。

我们来尝试使用一下刚刚定义的层：

```python
model = nn.Sequential(
    MyLinear(4, 5),
    nn.BatchNorm1d(5),
    nn.ReLU(),
    nn.Dropout(0.3), 
    MyLinear(5, 3),
    nn.BatchNorm1d(3),
    nn.ReLU(),
    nn.Dropout(0.5), 
    MyLinear(3, 1),
)

# 随机生成一个输入 (batch_size=5, in_features=4)
x = torch.randn(5, 4)
output = model(x)
print(output.shape)  # (5, 1)
```

**不含参层的实现**

有些层是没有可训练参数的，比如激活函数层。它们只是对输入进行固定的数学变换。我们来自己写一个LeakyReLU层。

这个层不需要任何可训练的参数，但可能需要在初始化时传入一些超参数（比如 α）。我们可以这样实现：

```python
class LeakyReLUEfficient(nn.Module):
    def __init__(self, negative_slope=0.01):
        super().__init__()
        self.negative_slope = negative_slope  # 保存为属性，供 forward 使用
    
    def forward(self, x):
        # 使用 torch.maximum 实现 LeakyReLU
        return torch.maximum(self.negative_slope * x, x)
```

可以看到，我们在定义模型的时候不需要写`backward`了，PyTorch 的**自动微分机制**会帮我们搞定一切。只要 `forward` 中的操作都是由 PyTorch 的张量运算组成（比如 `matmul`、`+`、`maximum` 等），Autograd 就会自动构建计算图，并在反向传播时自动计算梯度。我们完全不需要手动编写 `backward` 函数。

### 12.2 模型参数初始化方法

什么是权重初始值，为什么权重初始值的不同会对训练结果有较大影响？如果你还不清楚这些，请回看《深度学习入门：基于Python的理论与实现》这本书的6.2小节，这里写的非常详细。

PyTorch 在 `torch.nn.init` 模块中提供了很多初始化方法。下面介绍几种最常用的。

**零初始化 (Zeros)**：

```python
nn.init.zeros_(tensor)
```

将参数全部初始化为 0。这种方法虽然简单，但在神经网络中**基本不用**。为什么？因为如果所有神经元初始权重相同，那么在前向传播中它们会计算相同的输出，反向传播时梯度也相同，导致所有神经元同步更新，学不到多样化的特征。这叫做**对称性问题**。偏置有时可以初始化为 0，但权重绝对不能全零。

**随机初始化 (Random)：**

```python
# 均匀分布初始化
nn.init.uniform_(tensor, a=0.0, b=1.0)

# 正态分布初始化
nn.init.normal_(tensor, mean=0.0, std=1.0)
```

用小的随机数初始化是打破对称性的基本方法。通常我们从均匀分布或正态分布中采样小数值（例如 `std=0.01`）。但要注意，如果分布的范围选择不当，可能导致梯度消失或爆炸。比如，如果标准差太大，激活值可能过大，进入激活函数的饱和区；如果太小，梯度也可能过小。因此，我们需要更精细的初始化策略。

**Xavier 初始化：**

Xavier 初始化由 Glorot 和 Bengio 在 2010 年提出，旨在让每一层的输入和输出的方差保持一致，从而缓解梯度消失或爆炸。它适用于**tanh、sigmoid**这类**饱和激活函数**。

```python
# 均匀分布 Xavier
nn.init.xavier_uniform_(tensor, gain=1.0)

# 正态分布 Xavier
nn.init.xavier_normal_(tensor, gain=1.0)
```

**Kaiming 初始化 (He 初始化)：**

Kaiming He 等人针对 ReLU 及其变体（如 LeakyReLU）设计了这种初始化方法。因为 ReLU 会将一半的神经元置零，导致方差发生变化，所以需要专门处理。Kaiming 初始化在如今使用 ReLU 的深度网络中非常流行。

```python
# 均匀分布 Kaiming (适用于 ReLU)
nn.init.kaiming_uniform_(tensor, a=0, mode='fan_in', nonlinearity='relu')

# 正态分布 Kaiming
nn.init.kaiming_normal_(tensor, a=0, mode='fan_in', nonlinearity='relu')
```

除了以上介绍的常用方法，还有很多其他初始化技术，但大多数只需要了解即可。在实际使用中，如果不太确定该选哪一种，参数该如何设置，可以直接问问AI，比如“我的模型适合哪种权重初始化方法？”，它会根据你的具体情况给出定制化的建议。

### 12.3 自定义模型

在上一节中，我们学会了如何自定义一个神经网络层，就像制作了一块乐高积木。现在，我们要把这些积木拼起来，搭建一个完整的模型。在 PyTorch 中，自定义模型的方式和自定义层几乎完全一样——都是通过继承 `nn.Module` 类来实现。实际上，**模型本身也是一个 `Module`**，只不过它通常由多个子模块（层）组合而成。

你可能已经注意到，`nn.Module` 这个名字并没有区分“层”和“模型”。这是因为在 PyTorch 的设计哲学中，它们本质上是一样的：都是可组合的模块。一个简单的线性层是一个 `Module`，一个由几十个层组成的深度神经网络也是一个 `Module`。这种统一的设计让我们可以像搭积木一样，从小模块构建出大模块，再从大模块构建出更大的模型，层层嵌套，非常灵活。

接下来，我们将通过一个带有**残差连接**的全连接神经网络，来演示如何自定义模型。残差连接（Residual Connection）是现代深度网络（如 ResNet）的核心思想，它通过将输入直接加到输出上，缓解了深层网络中的梯度消失和退化问题。

首先，我们定义一个**残差块**（Residual Block）。它由两个全连接层组成，并在最后加上跳跃连接（shortcut connection），将输入与变换后的输出相加。

```python
class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        """
        初始化残差块
        hidden_dim (int): 隐藏层特征维度（输入输出维度相同，保证残差相加）
        dropout (float): Dropout丢弃率，默认0.1，用于防止过拟合
        """
        super().__init__()
        
        # 定义残差分支的主干网络（两层全连接 + 归一化 + 激活 + 正则化）
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 保存输入作为残差（shortcut连接）
        residual = x
        # 通过主干网络计算变换后的特征
        out = self.net(x)
        # 残差连接：将输入与变换后的输出相加（要求维度一致）
        out += residual
        # 相加后再应用激活函数，输出最终结果
        return torch.relu(out)
```

在这个块中，输入 `x` 先经过两个全连接层（中间有 ReLU），然后将第二个全连接层的输出与原始输入相加，最后再通过一个 ReLU 激活。这样，梯度就可以直接通过加法操作回传，训练更深的网络变得更容易。

有了残差块，我们就可以像搭积木一样把它们堆叠起来，构建一个更深的网络。下面的 `ResNetMLP` 类就是一个由多个残差块组成的多层感知机（MLP）风格模型。

```python
class ResNetMLP(nn.Module):
    def __init__(self, 
                 input_dim: int,      # 输入特征的维度
                 hidden_dim: int,     # 隐藏层特征维度
                 num_classes: int,    # 分类任务的类别数量
                 num_blocks: int = 3, # 残差块堆叠数量，控制网络深度
                 dropout: float = .1  # Dropout比率，正则化强度
                ):
        super().__init__()
        
        # === 1. 输入投影层：将任意输入维度映射到统一hidden_dim ===
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),      
            nn.BatchNorm1d(hidden_dim),         
            nn.ReLU()                 
        )
        
        # === 2. 残差块堆叠：构建深层网络的核心 ===
        # 使用列表推导式动态创建多个相同的残差块，并用 nn.Sequential 封装
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim, dropout) for _ in range(num_blocks)]
        )
        
        # === 3. 输出头：将隐藏特征映射到类别空间 ===
        # 注意：此处输出logits，后续配合CrossEntropyLoss使用
        self.head = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Step 1: 输入特征投影到隐藏空间
        x = self.input_proj(x)
        # Step 2: 通过多个残差块进行特征提取与非线性变换
        x = self.res_blocks(x)
        # Step 3: 输出层映射到类别分数（logits）
        return self.head(x)
```

现在，让我们来实例化并使用这个模型。假设我们有一个批大小为 64、每个样本特征维度为 128 的输入数据，希望将其分为 10 个类别。

```python
# 创建随机输入数据（模拟一个 batch）
input_data = torch.randn(64, 128)  # 形状: (batch_size, input_dim)

# 实例化模型
model = ResNetMLP(
    input_dim=128,      # 输入特征维度
    hidden_dim=240,     # 隐藏层维度（可调）
    num_classes=10,     # 输出类别数
    num_blocks=3,       # 堆叠 3 个残差块
    dropout=0.2         # Dropout 比率
)

# 前向传播，得到输出 logits
output_data = model(input_data)

print(output_data.shape)  # 输出形状应为 (64, 10)
```

运行上述代码，你会得到形状为 `(64, 10)` 的输出张量，每一行对应一个样本在 10 个类别上的 logits。接下来，你可以将这些 logits 传入损失函数（如 `nn.CrossEntropyLoss`）进行训练。这就是下一章要讲解的内容了。
