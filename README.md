# NLP 2026 秋季 作业一：文本分类（NYT）

刘迪乘 2312810 · 实验报告：[`hw1/report/main.pdf`](hw1/report/main.pdf)

在 NYT 三类新闻（sports / politics / business）上比较词袋、静态词向量和 BERT，在测试集上报告 Accuracy 与 Macro-F1。

## 目录结构

```
hw1/
  code/
    dataloader.py   # 读取 nyt.csv，按标签分层随机划分 80/10/10（seed=42）；评测与结果保存工具
    BoW.py          # Task 1：Binary BoW / Word Frequency（另附 TF-IDF）+ Logistic Regression
    word2Vec.py     # Task 2：GloVe / Word2Vec(AG News) / Word2Vec(NYT 训练集) 平均词向量 + LR
    bert.py         # Task 3：bert-base-uncased 微调（max_length=64，3 epochs）
    stats.py        # 数据集统计：划分规模、类别分布、文档长度、各长度下的截断篇数
    summarize.py    # 汇总 results/*.json，生成 results/summary.md 与错误样例
  run_all.sh        # 一键复现全部结果（含额外实验）
  results/          # 每次运行的指标（JSON）与汇总表 summary.md
  report/           # 实验报告 main.tex / main.pdf
  dataset/          # 数据（不入库，需自行放置，见下）
```

## 环境

- Python 3.12，依赖由 uv 管理（`pyproject.toml` / `uv.lock`）。
- PyTorch 从 CUDA 12.8 源安装，需要支持 CUDA 12.8 的 NVIDIA 驱动。Task 3 需要 GPU；Task 1、2 只用 CPU。
- 实验硬件：NVIDIA H800 80GB（每次 BERT 运行使用 1 张），Intel Xeon Platinum 8468V。
- 实测版本：torch 2.11.0+cu128、transformers 5.17.0、scikit-learn 1.9.1、gensim 4.4.0、nltk 3.10.3、numpy 2.5.3。

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh                 # 安装 uv
uv sync                                                         # 按 uv.lock 创建 .venv
uv run python -c "import nltk; nltk.download('punkt_tab')"      # word_tokenize 需要的分词数据
```

脚本在首次运行时也会自动检查并下载 `punkt_tab`。BERT 权重（`bert-base-uncased`，即 `google-bert/bert-base-uncased`）首次运行时从 Hugging Face 自动下载。

## 数据

把以下文件放到 `hw1/dataset/`，文件名必须一致：

| 文件 | 来源 | 内容 |
|---|---|---|
| `nyt.csv` | 课程提供的 HW-1 数据集 | 列 `text`, `label`，11519 篇，已小写 |
| `ag.csv` | 课程提供的 HW-1 数据集（AG News） | 列 `text`，90000 篇，已小写 |
| `glove.6B.100d.txt` | <https://nlp.stanford.edu/data/glove.6B.zip> 解压 | GloVe 6B 100 维 |

`word2Vec.py --train` 会把训练好的模型写到 `hw1/dataset/{ag,nyt}_seed{SEED}.w2v`。

## 运行

在仓库根目录执行。Word2Vec 只有在单线程且固定 `PYTHONHASHSEED` 时才能逐位复现，所以先设置：

```bash
export PYTHONHASHSEED=0
```

| 任务 | 命令 | 结果文件 |
|---|---|---|
| 数据统计 | `uv run python hw1/code/stats.py` | `results/stats.json` |
| Task 1 Binary BoW | `uv run python hw1/code/BoW.py --method binary` | `results/bow_binary.json` |
| Task 1 Word Frequency | `uv run python hw1/code/BoW.py --method frequency` | `results/bow_frequency.json` |
| （额外）TF-IDF | `uv run python hw1/code/BoW.py --method tfidf` | `results/bow_tfidf.json` |
| （额外）调 C | `uv run python hw1/code/BoW.py --method binary --C 0.1` | `results/bow_binary_C0.1.json` |
| Task 2 GloVe | `uv run python hw1/code/word2Vec.py --method glove` | `results/emb_glove.json` |
| Task 2 Word2Vec (AG News) | `uv run python hw1/code/word2Vec.py --method ag --train --seed 42` | `results/emb_ag_seed42.json` |
| Task 2 Word2Vec (NYT) | `uv run python hw1/code/word2Vec.py --method nyt --train --seed 42` | `results/emb_nyt_seed42.json` |
| （额外）NYT 全部正文训练 | `uv run python hw1/code/word2Vec.py --method nyt --train --nyt_all_text --seed 42` | `results/emb_nyt_all_seed42.json` |
| Task 3 BERT | `CUDA_VISIBLE_DEVICES=0 uv run python hw1/code/bert.py --max_length 64 --seed 42` | `results/bert_len64_seed42.json` |
| 汇总 | `uv run python hw1/code/summarize.py` | `results/summary.md`、`results/error_examples.json` |

- NYT 的 Word2Vec 只用训练集正文训练，验证集和测试集文本不参与。
- 已有 `.w2v` 模型时，去掉 `--train` 即可直接评测。
- `bert.py` 训练满 3 个 epoch，同时报告末轮模型和验证集 Macro-F1 最优 epoch 的测试结果；`--seed` 只影响训练，数据划分始终使用 seed 42。
- 每个 JSON 包含验证集与测试集的 Accuracy、Macro-F1、逐类 P/R/F1、混淆矩阵（行为真实标签，按 business/politics/sports 排序）和测试集预测。

一键复现全部结果（3 个种子 × Word2Vec，8 个长度 × 3 个种子的 BERT，以及调 C 实验）：

```bash
GPU=0 bash hw1/run_all.sh        # 可用 PYTHON=".venv/bin/python" 替换解释器
```

在 H800 上，BERT 长度 64 训练 3 个 epoch 约 1 分钟，长度 512 约 3.5 分钟；三种词袋各 20–110 秒。

## 预期结果（NYT 测试集，1152 篇）

Word2Vec 与 BERT 为种子 42/43/44 的均值 ± 样本标准差；其余方法是确定性的，只运行一次。

| 任务 | 方法 | Accuracy | Macro-F1 |
|---|---|---|---|
| 1 | Binary BoW | 0.9896 | 0.9741 |
| 1 | Word Frequency | 0.9913 | 0.9779 |
| 2 | GloVe 6B 100d | 0.9852 | 0.9646 |
| 2 | Word2Vec（AG News） | 0.9760 ± 0.0005 | 0.9415 ± 0.0016 |
| 2 | Word2Vec（NYT 训练集） | 0.9806 ± 0.0018 | 0.9524 ± 0.0037 |
| 3 | BERT，max_length 64，第 3 个 epoch | 0.9826 ± 0.0009 | 0.9615 ± 0.0006 |
| 3 | BERT，max_length 64，验证集最优 epoch | 0.9826 ± 0.0009 | 0.9611 ± 0.0012 |
| 额外 | TF-IDF | 0.9896 | 0.9751 |
| 额外 | Word2Vec（NYT 全部正文，含测试文本） | 0.9818 ± 0.0017 | 0.9546 ± 0.0042 |
| 额外 | BERT，max_length 512，第 3 个 epoch | 0.9931 ± 0.0009 | 0.9838 ± 0.0018 |

完整结果（逐类 F1、长度扫描、调 C、词向量 OOV 率）见 [`hw1/results/summary.md`](hw1/results/summary.md)。GPU 浮点运算的非确定性可能使 BERT 结果有极小波动。

## 实验设置摘要

- 划分：NYT 按标签分层随机划分 80/10/10（train 9215 / val 1152 / test 1152），seed=42，所有实验共用。
- Logistic Regression：saga，L2 正则，C=1.0，max_iter=10000，random_state=42；主结果不调参。
- Word2Vec：skip-gram，100 维，window 5，min_count 5，negative 5，5 epochs，workers=1。
- BERT：AdamW，lr 2e-5，batch 32，weight decay 0.01，10% 线性预热，梯度裁剪 1.0，3 epochs。
