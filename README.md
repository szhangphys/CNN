# CNN
a record about the 2017 Deep Learning course project at Peking university


基于双β衰变实验的数据特征，把它作为有监督的分类问题，通过卷积神经网络，实现对探测器事例类别的预测。此外，尝试在数据集上使
用迁移学习，获得了不错的结果。并且预期当信号维度信息更多时，迁移学习更加明显的体现出其优势。

数据通过CERN公布在Kaggle上的开放途径下载得到。数据集由14051775个探测器信号构成，每个信号对应球型探测器表面探测到一个光子，带有探测粒子的空间辐角和时间信息，共计8万次事例。

事件中信号来自于两种途径：粒子在闪烁体中超光速引起的切伦科夫效应释放的切伦科夫光子；与闪烁体探测器中电子相互作用产生的各向同性的闪烁体光子。
双β衰变和太阳中微子通过切伦科夫辐射产生光子的分布截然不同，理想情况下，双β衰变由于释放两个电子，两个电子释放方向相反的两个锥状光子束，而中微子释放的切伦科夫光则只有一个方向。

基于双β衰变与中微子事例的特征，我们以θ-φ为平面，对时间量做统计,按Bin区分时间(当作颜色通道处理)，双β衰变信号具有显著的双轨迹特征的空间分布，而中微子产生的切伦科夫辐射光信号则是单轨的。

In this paper, based on the data features of double β decay, we treat it as a supervised classification problem, and train it with CNNS to achieve the classification prediction of detector cases. In addition, we tried to use transfer learning on the data set and got good results.
