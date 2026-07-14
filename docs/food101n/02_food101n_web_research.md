# Food101N Phase 2 Web Research

日期：2026-07-14

## 搜索与访问说明

本阶段使用官方项目页、GitHub、arXiv、OpenReview、OpenAlex 和可访问 README/源码进行核查。Google Scholar 页面返回了可见 HTML，但自动化结果解析不稳定；Semantic Scholar API 返回 429 限流。因此 citation 网络记录以 OpenAlex、OpenReview、arXiv、项目页和 GitHub 为主。未虚构 Google Scholar 或 Semantic Scholar 结果。

## Food101N 官方来源

| 项目 | 结论 | URL |
|---|---|---|
| 官方数据页 | Food-101N 由 CleanNet/CVPR 2018 引入，约 310,009 张图、101 类、估计约 20% label noise | https://kuanghuei.github.io/Food-101N/ |
| 官方下载 | 下载入口指向 Kaggle `kuanghueilee/food-101n`，需用户同意非商业研究条款 | https://www.kaggle.com/datasets/kuanghueilee/food-101n |
| CleanNet 项目页 | CleanNet 项目页链接 paper、dataset、code | https://kuanghuei.github.io/CleanNetProject/ |
| CleanNet 论文 | arXiv 1711.07131；CVPR 2018 | https://arxiv.org/abs/1711.07131 |
| CleanNet 代码 | TensorFlow 代码；README 说明 CleanNet 使用 feature TSV，不是 PyTorch Dataset | https://github.com/kuanghuei/clean-net |

## Food101N 与 Food-101 区别

| 数据集 | 结论 | URL |
|---|---|---|
| Food-101N | noisy web-crawled food images，101 类，约 310k 张；每张图有 noisy class label，部分图有 verification label | https://kuanghuei.github.io/Food-101N/ |
| Food-101 | ETH Zurich 数据集，101 类、101,000 张；每类 750 train、250 manually reviewed test | https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/ |
| clean test | Food-101N 官方分类任务直接采用 Food-101 的 test set 评估 | https://kuanghuei.github.io/Food-101N/ |

Food101N 的训练标签是 noisy class label。verification label 是“该 noisy class label 是否正确”的二值人工验证标签，不等价于每张图的 clean class label。Food101N 官方分类评估应使用 Food-101 clean test labels。

## 官方数据格式线索

官方 Food101N 页面说明：

1. 每张图有 noisy class label。
2. 52,868 张训练图有 verification labels。
3. 4,741 张验证图有 verification labels。
4. 分类任务测试集来自 Food-101 test set。

Cleanlab benchmark README 记录了 Kaggle 解压后的常见路径：

```text
Food-101N_release.zip
Food-101N_release/train
Food-101N_release/meta/verified_train.tsv
```

来源： https://github.com/cleanlab/label-error-detection-benchmarks

CleanNet 代码 README 说明其模型输入是 TSV feature 文件，列可包含 sample key、image url、class name、verification label 和 feature。该格式用于 CleanNet 特征训练，不是本项目直接需要的图片 Dataset 格式。来源： https://github.com/kuanghuei/clean-net

## Food-101 类别映射

torchvision `Food101` 实现使用 `meta/train.json` 或 `meta/test.json`，读取 metadata key 并 `sorted(metadata.keys())` 得到类别顺序，再映射到 0..100。来源：

https://github.com/pytorch/vision/blob/main/torchvision/datasets/food101.py

为了让 Food101N train 和 Food-101 test 共享类别映射，本实现采用稳定排序的类别名列表，并允许用户通过 `food101n_classes_file` 显式指定类别文件。

## PuriDivER 公开 Food-101N continual split

PuriDivER 是 CVPR 2022 “Online Continual Learning on a Contaminated Data Stream with Blurry Task Boundaries”的官方 PyTorch 实现。README 说明其真实噪声数据集包括 mini-WebVision 和 Food-101N，并给出 `Food-101N` 数据目录格式和 `blurry10` 设置。来源：

https://github.com/clovaai/puridiver

PuriDivER 仓库提供 `tasks/Food-101N/*.json`。抽样查看 raw JSON，字段包括：

```text
file_name
verification_label
klass
label
```

示例文件：

https://github.com/clovaai/puridiver/tree/master/tasks/Food-101N

许可证：PuriDivER 为 GPL-3.0。不能直接复制其实现代码；本项目只兼容其公开 JSON 格式。

## Noise-Robust CIL 相关论文和实现

| 名称 | 年份 | 关系 | URL |
|---|---:|---|---|
| CleanNet | 2018 | 引入 Food101N；分类任务使用 Food-101 test | https://arxiv.org/abs/1711.07131 |
| PuriDivER | 2022 | 在线 continual learning + noisy/blurry stream，使用 Food-101N | https://arxiv.org/abs/2203.15355 |
| May the Forgetting Be with You: Alternate Replay for Learning with Noisy Labels | 2024 | AER/ABS 相关论文；OpenReview 页面可访问 | https://openreview.net/forum?id=AgCz44ebFe |
| BMVC 2024 Paper 680 PDF | 2024 | 与 AER/ABS 公开论文页面相关；当前环境缺 `pdftotext`，未自动抽取 PDF 表格 | https://bmva-archive.org.uk/bmvc/2024/papers/Paper_680/paper.pdf |

OpenAlex 搜索结果显示：

1. CleanNet 2018 记录被引约 485 次：https://openalex.org/W2962762068
2. `Food-101N continual learning label noise` 检索返回 PuriDivER 2022 与 AER 2024：https://openalex.org/W4312416777 ，https://openalex.org/W4402702526

## 候选实现比较

| 来源 | 可用内容 | 许可证 | 是否采用 |
|---|---|---|---|
| Food101N 官方页 | 数据语义、样本规模、评估协议、下载入口 | 数据有非商业研究条款 | 采用为数据语义主依据 |
| CleanNet GitHub | feature TSV 处理和 CleanNet 训练说明 | MSR-LA Full Rights License | 只参考数据字段说明，不复制代码 |
| torchvision Food101 | Food-101 class mapping 与 test metadata 读取方式 | BSD-style/PyTorch 项目许可 | 参考类别排序思想，不复制实现 |
| PuriDivER GitHub | Food-101N continual task JSON、blurry10 协议线索 | GPL-3.0 | 只兼容 JSON 字段，不复制代码 |
| cleanlab benchmark | Kaggle 解压路径、`verified_train.tsv` 线索 | Apache-2.0 | 参考路径线索，不复制代码 |
| Mammoth upstream | 未找到官方 `seq_food101n.py` | MIT | 没有可用 Food101N 实现 |

## 最终参考来源

1. Food101N 官方数据页： https://kuanghuei.github.io/Food-101N/
2. Food-101 官方数据页： https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/
3. CleanNet 项目页： https://kuanghuei.github.io/CleanNetProject/
4. CleanNet 代码： https://github.com/kuanghuei/clean-net
5. torchvision Food101： https://github.com/pytorch/vision/blob/main/torchvision/datasets/food101.py
6. PuriDivER： https://github.com/clovaai/puridiver
7. PuriDivER task JSON： https://github.com/clovaai/puridiver/tree/master/tasks/Food-101N
8. Cleanlab benchmark： https://github.com/cleanlab/label-error-detection-benchmarks
9. AER OpenReview： https://openreview.net/forum?id=AgCz44ebFe

## 尚未解决的问题

1. CleanNet/Food101N 官方没有公开用于 class-incremental protocol 的 class order。
2. Food101N 官方 classification train metadata 的完整文件名需用户本地数据确认；Kaggle 未在当前环境登录下载。
3. Food101N train 只有 noisy class label 和部分 verification label；没有完整 clean train class label。
4. AER/BMVC PDF 的 Food-101N 具体 transform 和超参未能用本机 PDF 文本工具自动抽取，需要后续人工核对论文 PDF 或补装 PDF 工具。
5. 本机没有真实 Food101N 数据，无法验证官方样本数。
