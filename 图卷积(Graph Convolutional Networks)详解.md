# 图卷积 (Graph Convolutional Networks) 详解
> 
> 参考 [Kipf and Welling.  Semi-Supervised Classification with Graph Convolutional Networks. ICLR. 2017](https://arxiv.org/abs/1609.02907)
<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>
# 1. 拉普拉斯特征映射与谱聚类

由于后续的GCN模型中用到了正则化的拉普拉斯矩阵，所以我对这一部分的知识进行了补充。 内容主要参考 [July博文](https://blog.csdn.net/v_july_v/article/details/40738211) 
### 1.1 谱聚类
聚类是图论中的一个经典问题，可以理解成图的分割问题，即把图的顶点集分割为不相交的子集形成两个或多个子图。最终想要达到的效果是： 分割后的若干个子图内部边的权重尽可能高，而子图之间边的权值尽可能低。
子图切割有许多不同的方法，例如: 
1. `RatioCut`
2. `Normalized Cut`
3. `转化为SVD可解的问题`

接下来，首先介绍`RatioCut`方法是如何做到子图切割的。
拿到一个图，我们如何进行切割才能得到最优的结果呢？显然在切割的时候，需要切断图中的一些边，那么可以想到用边上的权值作为度量，尽可能去切那些权值比较小的边，因为权值较小的边意味着两个节点之间的关联(相似性)较弱。我们以"被切断的边的权值之和最小"作为目标函数，将其形式化的表示如下：

------

设![A_1,...,A_k](https://math.jianshu.com/math?formula=A_1%2C...%2CA_k)为图划分子集，为了让分割的效果最好，便要最小化下面的损失函数:
![cut(A_1,...,A_k) := \frac12\sum_{i=1}^k W(A_i, \overline A_i)](https://math.jianshu.com/math?formula=cut(A_1%2C...%2CA_k)%20%3A%3D%20%5Cfrac12%5Csum_%7Bi%3D1%7D%5Ek%20W(A_i%2C%20%5Coverline%20A_i))

其中![k](https://math.jianshu.com/math?formula=k)表示分割子图数量，![A_i](https://math.jianshu.com/math?formula=A_i)表示第![i](https://math.jianshu.com/math?formula=i)个分割子图，![\overline A_i](https://math.jianshu.com/math?formula=%5Coverline%20A_i)表示![A_i](https://math.jianshu.com/math?formula=A_i)的补集，![W(A_i,\overline A_i)](https://math.jianshu.com/math?formula=W(A_i%2C%5Coverline%20A_i))表示分割子图![A_i](https://math.jianshu.com/math?formula=A_i)与![\overline A_i](https://math.jianshu.com/math?formula=%5Coverline%20A_i)之间的所有边的权重之和。

------

然而，这种方法通常会导致一些意外的结果。由于损失函数没有限制，该方法往往会将图分成一个点和其余n-1个点。如下图所示。 

![image](http://r.photo.store.qq.com/psb?/V14RoQOQ2suUoC/INfTHW0hNm7Lq0owF4iAn*OdFLHf92*Ucob4Lcf7vbY!/r/dMEAAAAAAAAA&ynotemdtimestamp=1545279693625)
为了克服这一问题，我们可以在损失函数中加入分割子图Ai的大小这一限制，使得损失函数在最小化的同时保证各个分割子图具有合理的大小。公式表示如下：

------

![RatioCut(A_1,...,A_k) := \frac12\sum_{i=1}^k\frac {W(A_i, \overline A_i)} {|A_i|} = \sum_{i=1}^k\frac {cut(A_i, \overline A_i)} {|A_i|}](https://math.jianshu.com/math?formula=RatioCut(A_1%2C...%2CA_k)%20%3A%3D%20%5Cfrac12%5Csum_%7Bi%3D1%7D%5Ek%5Cfrac%20%7BW(A_i%2C%20%5Coverline%20A_i)%7D%20%7B%7CA_i%7C%7D%20%3D%20%5Csum_%7Bi%3D1%7D%5Ek%5Cfrac%20%7Bcut(A_i%2C%20%5Coverline%20A_i)%7D%20%7B%7CA_i%7C%7D)

其中![|A_i|](https://math.jianshu.com/math?formula=%7CA_i%7C)表示分割子图![A_i](https://math.jianshu.com/math?formula=A_i)中包含的顶点数。

------

改进后的方法就是上面提到的`RatioCut`方法，后面的两类方法不多做介绍。接下来我们将要探讨拉普拉斯矩阵与谱聚类的关系，更详细一点，拉普拉斯矩阵与`RatioCut`方法的关联。

### 1.2 拉普拉斯矩阵

首先介绍拉普拉斯矩阵的定义和相关性质(后文会用到)。

#### 1.2.1 定义

拉普拉斯矩阵式表示图的一种矩阵，给定一个图 $G = (V, E)$，其拉普拉斯矩阵的定义如下：

 									$L = D - A$

其中$D$为图的度矩阵，$A$为图的邻接矩阵。 举个例子，给定如下的图：
![image](http://r.photo.store.qq.com/psb?/V14RoQOQ2suUoC/*2t*Mz8iBKrtyqCZFxX8UnTOCJTK7.Mg5QFYzyydTtI!/r/dLwAAAAAAAAA&ynotemdtimestamp=1545279693625)  	
邻接矩阵`A`表示为：
![image](http://r.photo.store.qq.com/psb?/V14RoQOQ2suUoC/glFhogklUHho4UgrbyyQE9i53KlJtDSBhOcra81Oa88!/r/dDQBAAAAAAAA&ynotemdtimestamp=1545279693625)
计算图中每个节点的度，并将它们放在主对角线上，组成一个对角矩阵，记为度矩阵`D`，如下所示：
![image](http://r.photo.store.qq.com/psb?/V14RoQOQ2suUoC/ljcOjAn32vnTRWt4JgpTy1sEZ3iWZlAVpyvA1Jl.9tk!/r/dL0AAAAAAAAA&ynotemdtimestamp=1545279693625)
根据拉普拉斯矩阵的定义，可以计算出拉普拉斯矩阵`L`：
![image](http://r.photo.store.qq.com/psb?/V14RoQOQ2suUoC/0z*aTIE4*kO7SGHIOi4eWhGytfceuo4QNw97tV79*Nc!/r/dLgAAAAAAAAA&ynotemdtimestamp=1545279693625)

#### 1.2.2 性质

拉普拉斯矩阵L具有如下性质：

1. $L$是对称半正定矩阵
2. $L$的最小特征值为0，相应的最小特征向量为![\vec{e}](https://math.jianshu.com/math?formula=%5Cvec%7Be%7D)
3. $L$有n个非负实特征值

![0 = \lambda_1 \leq ... \leq \lambda_n](https://math.jianshu.com/math?formula=0%20%3D%20%5Clambda_1%20%5Cleq%20...%20%5Cleq%20%5Clambda_n)

1. 对于任何一个实向量![f=f(f_1,...,f_n)'\in \mathbb R^n](https://math.jianshu.com/math?formula=f%3Df(f_1%2C...%2Cf_n)%27%5Cin%20%5Cmathbb%20R%5En)，有以下式子成立：

![f'Lf = \frac12\sum_{i,j=1}^N w_{ij}(f_i-f_j)^2](https://math.jianshu.com/math?formula=f%27Lf%20%3D%20%5Cfrac12%5Csum_%7Bi%2Cj%3D1%7D%5EN%20w_%7Bij%7D(f_i-f_j)%5E2)

![where \space L = D-W, d_i = \sum_{j=1}^nw_{ij}, W(A,B):=\sum_{i\in A,j\in B}w_{ij}](https://math.jianshu.com/math?formula=where%20%5Cspace%20L%20%3D%20D-W%2C%20d_i%20%3D%20%5Csum_%7Bj%3D1%7D%5Enw_%7Bij%7D%2C%20W(A%2CB)%3A%3D%5Csum_%7Bi%5Cin%20A%2Cj%5Cin%20B%7Dw_%7Bij%7D)

下面，给出上述结论的证明。

![f'Lf =f'Df-f'Wf= \sum_{i=1}^n d_{i}f_i^2-\sum_{i,j=1}^n f_{i}f_jw_{ij}\\ =\frac12(\sum_{i=1}^nd_if_i^2-2\sum_{i,j=1}^nf_if_jw_{ij}+\sum_{j=1}^nd_jf_j^2)\\ =\frac12\sum_{i,j=1}^nw_{ij}(f_i-f_j)^2](https://math.jianshu.com/math?formula=f%27Lf%20%3Df%27Df-f%27Wf%3D%20%5Csum_%7Bi%3D1%7D%5En%20d_%7Bi%7Df_i%5E2-%5Csum_%7Bi%2Cj%3D1%7D%5En%20f_%7Bi%7Df_jw_%7Bij%7D%5C%5C%20%3D%5Cfrac12(%5Csum_%7Bi%3D1%7D%5End_if_i%5E2-2%5Csum_%7Bi%2Cj%3D1%7D%5Enf_if_jw_%7Bij%7D%2B%5Csum_%7Bj%3D1%7D%5End_jf_j%5E2)%5C%5C%20%3D%5Cfrac12%5Csum_%7Bi%2Cj%3D1%7D%5Enw_%7Bij%7D(f_i-f_j)%5E2)

### 1.3 RatioCut与拉普拉斯矩阵的关联

回到上面的`RatioCut`方法，其优化式子表示如下：

![min_{A\subset V} \space RatioCut(A, \overline A)](https://math.jianshu.com/math?formula=min_%7BA%5Csubset%20V%7D%20%5Cspace%20RatioCut(A%2C%20%5Coverline%20A))

定义向量

$$f = (f_1,...,f_n)' \in R^n, and : f_i = \begin{cases} \sqrt{|\overline A|/|A|}&if & v_i \in A \\ -\sqrt{|A|/|\overline A|} & if & v_i \in \overline A \end{cases}$$

根据之前得到的拉普拉斯矩阵的性质，有

$$f'Lf = \frac12\sum_{i,j=1}^Nw_{ij}(f_i-f_j)^2$$

将\\(f_i\\)带入上式，得：

$$ f'Lf = \frac12\sum_{i,j=1}^Nw_{ij}(f_i-f_j)^2 $$

$$=\frac12\sum_{i\in A,j\in\overline A}w_{ij}(\sqrt{\frac {|\overline A|}{|A|}}+\sqrt{\frac {|A|}{|\overline A|}})^2 + \sum_{i\in \overline A,j\in A}w_{ij}(-\sqrt{\frac {|\overline A|}{|A|}}-\sqrt{\frac {|A|}{|\overline A|}})^2 $$

$$=cut(A,\overline A)(\frac{|\overline A|}{|A|} + \frac{|A|}{|\overline A|}+2) $$           

$$=cut(A,\overline A)(\frac{|A|+|\overline A|}{|A|} + \frac{|A|+|\overline A|}{|\overline A|})   \   $$       

$$=|V| RatioCut(A,\overline A) $$      

可以发现，我们从$f'Lf$推导出了`RatioCut`。也就是说拉普拉斯矩阵L和我们要优化的目标`RatioCut`有着密切的联系。由于这里$|V|$是一个常量，所以最小化`RatioCut`函数，等价于最小化$f'Lf$。
同时由于单位向量$\vec e$的各个元素全为1，所以可以直接展开得到约束条件：

$$||f||^2=\sum f_i^2 = n \space and \space f'\vec e = \sum f_i = 0$$

如此一来，我们可以将目标函数写成如下形式：

$$\min_{f\in \mathbb R^n} f'Lf$$

$$s.t. \space f \perp \vec e, ||f|| = \sqrt n$$

根据特征值和特征向量的定义，我们可以对上面的优化目标进行转换。假设 $Lf = \lambda f$，即认为$\lambda$和$f$分别为$L$的特征值和对应的特征向量。两边同时左乘$f'$，可以得到$f'Lf=\lambda f'f=\lambda n$
因此，最小化$f'Lf$等价于最小化$\lambda$。所以接下来我们只需要找到$L$的最小特征值$λ$及其对应的特征向量即可。
然而，拉普拉斯矩阵的最小特征值为0，对应的特征向量正好为$\vec e$，不满足约束条件。根据`Rayleigh-Ritz`理论([`A Tutorial on Spectral Clustering`](http://engr.case.edu/ray_soumya/mlrg/Luxburg07_tutorial_spectral_clustering.pdf))，我们可以取第二小的特征值。
更进一步，如果我们取拉普拉斯矩阵的前K个特征向量，进行K-means聚类，便从二聚类扩展到了K聚类。
这样，就将离散求解$f=f(f_1,...,f_n)'\in \mathbb R^n$这一困难的NP问题转换成拉普拉斯矩阵特征值(向量)的问题，将离散的聚类问题松弛为连续的特征向量。最小的系列特征向量对应着图的最优系列划分，对特征向量进行划分，再离散回对应的类别即可。
关于拉普拉斯矩阵的理解：拉普拉斯矩阵实际上是将图中的节点进行了一次映射，这种映射增强了节点的聚集属性(cluster-properties)，这也解释了为什么我们无法通过邻接矩阵`AA`来做子图划分，而可以直接利用拉普拉斯矩阵的特征向量来完成(细节部分参考[`A Tutorial on Spectral Clustering`](http://engr.case.edu/ray_soumya/mlrg/Luxburg07_tutorial_spectral_clustering.pdf))。
此外，在图形学领域，为了保留像素之间的空间结构化信息，拉普拉斯矩阵还被用来做平滑(参考[`博客`](https://blog.csdn.net/bbbeoy/article/details/71249310))。

## 2. 图卷积的定义

介绍完基础的谱聚类背景知识，接来下开始介绍图卷积模型。目前大多数基于图结构的神经网络都遵从一个通用的结构，因此我们把这一类模型统称为图卷积网络。图卷积之所以被称为卷积是因为过滤器的参数通常会在图中的所有位置共享。这一类模型的目的是为了从图 $G = (V, E)$ 中学习一个关于信号/特征的函数，模型的输入输出表示如下。

------

输入：

- 图中每个节点的特征描述 $x_i$ ，组合成一个$N\times D$的特征矩阵$X$，其中$N$为节点数量，$D$为特征维数。
- 图的邻接矩阵表示为$A$

输出：

-节点层面的输出$Z$，$Z$是一个$N \times F$的特征矩阵，其中$F$为每个节点的输出特征维度

------

神经网络可以用一个非线性的函数来形式化表示：

$H^{(l+1)} = f(H^{(l)}, A)$

将这个函数对应到图卷积模型，这里的$H^{(0)}=X$，$H^{(L)}=Z$，$L$为神经网络的层数。

## 3. 图卷积模型详解

### 3.1 快速卷积

我们举一个简单的例子来描述GCN。将GCN模型的传播规则简化描述如下：

$H^{(l+1)} = \sigma(AH^{(l)}W^{l})$

其中$W^{(l)}$为神经网络第l层的权重矩阵，$\sigma (·)$为一个非线性的激活函数(例如ReLU函数)。 这里有两个限制，第一个在于上式乘以$A$代表着，对于每一个节点，我们将其邻居节点的所有特征向量进行了加和(sum up)，而没有考虑节点本身。这一限制可以通过给加一个$A$单位阵来解决(即强制自环)。此外，$A$矩阵是没有正则化的，因此乘以$A$会完全改变特征向量的尺度。这一限制可以通过给`AA`进行正则化，即使得$A$的所有行加和等于1，如$D^{-1}A$, 其中$D$为度矩阵。乘以$D^{-1}A$这一步骤所对应的物理意义是取相邻节点特征的平均值。实际上，在使用对称标准化时动力学会更为明显，例如$D^{-1/2}AD^{-1/2}$(参考[`Spectral Graph Theory`](https://people.orie.cornell.edu/dpw/orie6334/lecture7.pdf))。因为这不再是单纯取邻居节点的特征，而是。。。（待弄清楚）
由此可以得出[`Kipf&Welling(ICLR 2017)`](https://arxiv.org/abs/1609.02907)论文中的传播规则：

$f(H^{(l)}, A) = \sigma (D^{-1/2}\hat AD^{-1/2}H^{(l)}W^{(l)})$

### 3.2 谱图卷积

图卷积按映射域不同可以分为直接卷积和谱卷积两类，这篇问题是基于谱卷积的方法。在谱图卷积中，需要使用一个过滤器$g_{\theta}=diag(\theta)$，而原来的图数据在这里被替换成了频率信号$x \in \mathbb R^N$(对于每个节点而言是一个标量)。详细公式如下：

$g_{\theta} \bigotimes x= Ug_{\theta}U^{\tau}x$

其中$U$为正则化拉普拉斯矩阵$L=I_N-D^{-1/2}AD^{-1/2}=U\Lambda U^{\tau}$的特征向量矩阵，$U^{\tau}x$则表示对$x$进行傅里叶变换。这里滤波器$g_{\theta}$可以看作一个关于$L$特征值的函数，例如$g_\theta (\Lambda)$。但是计算量会非常大，其一是对`UU`做乘法时间复杂度很大；其二是如果数据很大，对$L$做特征分解也需要大量时间。因此作者利用了对切比雪夫多项式做截断来近似地估计$g_\theta (\Lambda)$。 公式如下：

$$g_{\theta'}(\Lambda) \approx \sum_{k=0}^K\theta_k'T_k(\hat \Lambda)$$

其中，$\hat \Lambda=\frac2{\lambda_{max}}\Lambda-I_N$。 $\lambda_{max}$为拉普拉斯矩阵$L$的最大特征值。$\theta'\in \mathbb R^K$为切比雪夫多项式的系数向量。切比雪夫多项式的递归定义为$T_k(x)=2xT_{k-1}(x)-T_{k-2}(x),with\space T_0(x)=1\space and\space T_1(x)=x$
(参考[Hammond et al.](https://arxiv.org/abs/0912.3848))
将上述式子结合可以得到谱卷积模型的卷积定义：

$g_{\theta'} \bigotimes x\approx \sum_{k=0}^K\theta_k'T_k(\hat L)x$

其中$\hat L=\frac2{\lambda_{max}}L-I_N$；由于这里截取了前K项，因此上式仅仅却决距离中心节点最大步长为K的节点(K阶邻域)。这就是[`Michaël et al,. NIPS 16`](https://arxiv.org/pdf/1606.09375.pdf/)的谱卷积工作。

### 3.3 Layer-wise线性模型

从上式可以看出，基于图卷积的神经网络模型可以通过叠加多个卷积层，每层加入一个逐点非线性(可以理解成ReLU激活函数)。论文中限制$K=1$，因此原函数可以视为一个关于拉普拉斯矩阵$L$的线性函数。
直观上可以预期模型在处理节点度分布非常广的图时，缓解对局部邻域结构的过拟合问题。此外，在计算资源受限的情况下，这种分层线性公式允许我们构建一些更深的网络模型，这一点在解决其他领域的问题上已经有所体现。 由于正则化拉普拉斯矩阵的特征值范围为[0, 2]，这里我们近似地认为$\lambda_{max} \approx 2$，将模型卷积公式改写为如下形式：

$g_{\theta'} \bigotimes x \approx {\theta_0 '}x + {\theta_1 '}(L-I_N)x = {\theta_0 '}x - {\theta_1 '}D^{-1/2}AD^{-1/2}x$

其中参数$\theta_0 '$和$\theta_1 '$为滤波器的参数，将在整个图的卷积过程中被共享。连续地应用这种形式的滤波器可以有效地卷积图中节点的`kk`阶邻域，其中`kk`是神经网络模型中连续滤波操作的次数或卷积层的个数。
在实际操作时，可以进一步限制参数的数量，在防止过拟合的同时减少每层的计算量。参数限制后的函数表示如下：

$g_{\theta'} \bigotimes x \approx \theta_0(I_N + D^{-1/2}AD^{-1/2})x$

其中，$\theta = \theta_0 ' = - \theta_1 '$。 注意到$I_N + D^{-1/2}AD^{-1/2}$特征值的范围是[0, 2]，重复进行卷积操作可能会导致数值不稳定以及梯度消失/爆炸等问题，为了解决这一问题，可以采用重整化方法：$I_N + D^{-1/2}AD^{-1/2} \rightarrow \hat D^{-1/2}\hat A \hat D^{-1/2}$ ，其中$\hat A = A + I_N$，$\hat D_{ii} = \sum_j \hat A_{ij}$。 将这个定义概括为一个信号表示：$X \in \mathbb R^{N \times C}$($C$为每个节点的特征维度)，过滤器(特征映射)$F$表示如下：

$Z = \hat D^{-1/2}\hat A \hat D^{-1/2}X \Theta$

其中$\Theta \in \mathbb R^{C \times F}$为滤波器参数矩阵，$Z \in \mathbb R^{N \times F}$为信号卷积矩阵。

## 4. 图卷积模型用于节点分类任务

上一章我们介绍了图上的信息传播模型$f(X,A)$，这一章将利用模型去完成节点分类任务。以往的模型在做这类问题时，通常会通过基于图的显示正则化将标签信息在图上做平滑，例如对损失函数添加图拉普拉斯正则项：

$\mathcal L = \mathcal L_0 + \lambda \mathcal L_{reg}$

$with \mathcal \space \space L_{reg}=\sum_{i,j}A_{ij}||f(X_i)-f(X_j)||^2 = f(X)^{\tau} \Delta f(X)  $

其中，$\mathcal L_0$表示图中标签样本的监督损失(`supervised loss`)，$\lambda$为权重因子，$X$为节点特征矩阵，$\Delta = D-A$表示无向图的未正则拉普拉斯矩阵。显然，这种损失函数的定义依赖于潜在的假设：`图中相互连接的节点可能共享相同的标签`。然而这种假设可能会限制模型的能力，尤其是当图中的边不一定编码了节点的相似性，而是携带了附加信息。例如citation network中的引用链接或知识图谱中的关系等。
基于这一考虑，论文构造模型时将节点特征矩阵$X$和描述底层图结构的邻接矩阵$A$作为独立的输入，进而调整模型。

### 4.1 简单模型

为了方便描述，这里介绍用于节点分类的两层GCN模型。首先，进行预处理操作计算$\hat A = \hat D^{-1/2}\hat A \hat D^{-1/2}$，则前馈模型可以简化描述为：

$Z = f(X, A) = softmax(\hat A\space \mathrm {ReLU}(\hat AXW^{0})W^{(1)})$

其中，$W^{(0)}\in \mathbb R^{C\times H}$为`input-to-hidden`权重矩阵，$H$为特征映射维度，$W^{(0)}\in \mathbb R^{C\times H}$为`hidden-to-output`权重矩阵。损失函数定义为：

$\mathcal L = -\sum_{l \in \mathcal Y_L} \sum_{f=1}^F Y_{lf}lnZ_{lf}$

其中$\mathcal Y_L$为有标签的节点索引。

### 4.2 实验

#### 4.2.1 数据集与参数设定

![image](http://r.photo.store.qq.com/psb?/V14RoQOQ2suUoC/s3MXfNyblmrjeCtqap32XD.Z3P9XusEALi9yLsvoIWk!/r/dL4AAAAAAAAA&ynotemdtimestamp=1545279693625)对于`Citation networks`，每个节点(文档)都有一个类别标签，在训练阶段作者对于每一类取了20个标签，但用到了所有的特征向量。 对于从知识图谱中抽取的数据集`NELL`，每个节点有61,278维的稀疏特征向量。对于此数据的分类任务，作者每个类仅取了1个标签。
测试集设置为一组1,000个标记样本。
对于超参数优化，作者设定`training iterations`为`200 epochs`，使用Adam优化器进行优化，学习率为0.01，`early stopping`窗口为10(连续10次`epochs`校验损失都没有减小则停止训练)

#### 4.2.2 代码解读

见[`github`](https://github.com/tools-only/Adversarial-attacks-on-GCNs/blob/master/gcn_base.ipynb)

#### 4.2.3 模型评估

与`baseline`的比较结果如下所示： ![image](http://r.photo.store.qq.com/psb?/V14RoQOQ2suUoC/hfhs*e1qqd7HyKfIPdqiOTma1CnYU3ki1DuQKBUjWxA!/r/dMMAAAAAAAAA&ynotemdtimestamp=1545279693625) 此外论文对比了上文介绍的几种传播模型，具体结果如下： ![image](http://r.photo.store.qq.com/psb?/V14RoQOQ2suUoC/rIyuOfX1SHnaLYlhrQHRr5S1zqrv.4FHHASdv6KsgGg!/r/dDQBAAAAAAAA&ynotemdtimestamp=1545279693625)结果表明`Renormalization trick`的效果最好。