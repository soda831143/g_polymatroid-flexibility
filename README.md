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
