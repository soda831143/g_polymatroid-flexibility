# Flexitroid

A repository for aggregating flexibility in populations of distributed energy resources (DERs): aggregate, disaggregate, optimize, and quantify uncertainty.

## Installation

```bash
pip install flexitroid

总结来说，Flexitroid代码库实现了Mukhi等人的第一篇论文 ("Exact Characterization of Aggregate Flexibility via Generalized Polymatroids") 中提出的理论框架：

用g-polymatroid（通过子模/超模函数 b 和 p）来精确表示各种DER的灵活性。
通过对这些函数求和来实现精确的聚合。
提供了在该聚合灵活性上进行线性规划（以及更复杂的QP、L-inf问题）的有效算法。
代码结构清晰，区分了设备层、聚合层和优化问题层。
它似乎没有直接包含Mukhi等人第二篇论文中关于分布鲁棒聚合的复杂内容（如Wasserstein距离、最坏情况总体构建），也没有包含Liu等人论文中的连续时间建模、仿射变换优化和DRCC。但README.md中提到可以“量化不确定性”，这可能暗示了未来扩展或存在未体现在这些核心文件中的相关模块，或者是指通过对参数进行采样分析不确定性的影响，而非像第二篇论文那样构建具有严格概率保证的分布鲁棒集合。