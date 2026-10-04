# /data/k

只使用 Qwen3.5-4B。环境是 `context_8` 里的 `swift`，不依赖已删除的 ms-swift 源码。

## 目录

```text
/data/k/
  generator/          当前要跑的：未微调生成器测试
    schema.py         按 v2-k 的 tables 取 DDL
    prompt.py         生成器提示词
    eval.py           调 8002 端口，和标准 SQL 做文本对比
    serve.sh          只启动 Qwen3.5-4B
  src/                旧的双模型 oracle 推理，先不动
  runs/generator-base/  本次测试的 predictions.jsonl 和 metrics.json
  log/  run/          服务日志和 pid
```

题库仍在 `/data/data/v2-k/QA_train.json`。这里不复制数据。

## 这次测什么

生成器输入是问题 + 标准表的 DDL。不测 Router，也不测知识检索。`--with-knowledge` 才会把已有的 `knowledge_ids` 拼进去。

文本完全匹配只作基线，不是比赛的执行分。

## 运行

```bash
bash /data/k/generator/serve.sh
conda activate context_8
python /data/k/generator/eval.py --limit 20
```

服务地址是 `http://127.0.0.1:8002/v1`。全量去掉 `--limit`。跨源另加 `--include-cross`。

## Router R0

标签来自 `/data/data/v2-k/QA_train.json`。代码在 `/data/k/generator/router/`。

```bash
python /data/k/generator/router/export_jsonl.py
bash /data/k/generator/router/train.sh
python /data/k/generator/router/eval.py
```

导出文件是 `/data/k/runs/router-r0/train.jsonl`，旁边有 `export_manifest.json`（含 sha256）。训练用 `context_8`、GPU 1、seed `20261001`。评测打在同一批 5830 条上，看的是是否学会映射。

## 生成实验

编号仍是 E01、E02、E03、E05。`/data/k/runs/experiments` 是第一版提示词的结果。PostgreSQL 和 Elasticsearch 的系统提示按标注写法改过之后，结果写到 `/data/k/runs/experiments-v1`，编号不变。MySQL 和 SQLite 的提示词没变，v1 沿用第一版里这两类的预测。

```bash
bash /data/k/experiments/run_v1.sh
```

测试集 `/data/raw/contest_8/QA_test.json` 只有问题和题号。目前能跑的是 E01，结果在 `/data/k/runs/experiments-test-v1`。E02、E03 要先用 Router 对这 400 题推理出数据源、库名、表名。E05 还要先检索知识点。

```bash
bash /data/k/experiments/run_test.sh
```
