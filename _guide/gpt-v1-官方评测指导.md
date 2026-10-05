# GPT-v1 官方评测指导

日期：2026-10-04。提交文件是 `/data/submit/gpt-v1/result.json`，400 题。下面的全对、得分、得分率是比赛官方测试集回传，用来给 `/data/k` 后续提交定对照，不代替本地执行评测。

## 结论

这版官方得分 **912.9091 / 1087（83.98%）**，全对 **340 / 400（85.00%）**。丢掉的 **174.0909** 分里，SQLite 占 144.5 分，knowledge 题型占 152 分。PostgreSQL 38 题和联邦 12 题在官方口径下都是满分，但这两类的判定系数是「预测查询执行结果非空」，拿到满分只说明查询返回了至少一行。

下一版要涨分，改 SQLite 的 knowledge 题。PostgreSQL 和联邦保持能返回结果即可，不要为了官方分去改这两类已经非空的查询。

## 计分

总分 = Σ(每题得分)。每题得分 = 该题满分 × 判定系数。满分由题型 `type` × 难度 `complexity` 查官方脚本 `SCORE_MATRIX`。该表不在本仓库，本文只使用官方回传的汇总数，不补格子。

判定系数按引擎分开：

| 引擎 | 判定系数 | 对本版结果的含义 |
|---|---|---|
| Elasticsearch | ID 集合匹配度 × 0.5 + 顺序匹配度 × 0.5，可取 0 到 1 之间的小数 | 50 题里 49 题系数为 1，剩下 1 题贡献了全部 1.0909 分的小数缺口 |
| PostgreSQL、联邦 | 预测查询执行结果非空则为 1，否则为 0 | 38 + 12 题系数为 1，只说明结果集非空 |
| MySQL、SQLite | 结果集与标准答案等价则为 1，否则为 0 | 12 道 MySQL、47 道 SQLite 系数为 0，每题整段满分都丢掉 |

全对率 85.00% 高于得分率 83.98%。未全对的 60 题更贵：59 道系数为 0 的题合计满分 173，均分 2.93；全部 400 题均分 2.72。所以同样错一题，knowledge 和 complex 比 simple、easy 掉分更多。

## 官方回传

总体：题数 400，全对 340，准确率 85.00%，得分 912.9091 / 1087，得分率 83.98%。

### 数据库类型

| 类型 | 全对 | 得分 | 得分率 | 丢掉 |
|---|---:|---:|---:|---:|
| Elasticsearch | 49 / 50 | 135.9091 / 137 | 99.20% | 1.0909 |
| 联邦 | 12 / 12 | 57 / 57 | 100.00% | 0 |
| MySQL | 88 / 100 | 236 / 264.5 | 89.22% | 28.5 |
| PostgreSQL | 38 / 38 | 59 / 59 | 100.00% | 0 |
| SQLite | 153 / 200 | 425 / 569.5 | 74.63% | 144.5 |

SQLite 占试卷满分的 569.5 / 1087 = 52.4%，占全部失分的 144.5 / 174.0909 = 83.0%。MySQL 占失分的 16.4%。Elasticsearch 占 0.6%。

### 题型

| type | 全对 | 得分 | 得分率 | 丢掉 |
|---|---:|---:|---:|---:|
| calculation | 7 / 7 | 33 / 33 | 100.00% | 0 |
| federation | 12 / 12 | 57 / 57 | 100.00% | 0 |
| geo | 29 / 31 | 123 / 132 | 93.18% | 9 |
| intermediate | 3 / 3 | 13.5 / 13.5 | 100.00% | 0 |
| knowledge | 178 / 226 | 531 / 683 | 77.75% | 152 |
| simple | 111 / 121 | 155.4091 / 168.5 | 92.23% | 13.0909 |

knowledge 占全部失分的 87.3%。simple 的 13.0909 分里，9 道结果集不等价的题合计 12 分，另外 1.0909 分是那道 Elasticsearch 部分分。geo 两道系数为 0，合计 9 分，均分 4.5。

### 难度

| complexity | 全对 | 得分 | 得分率 | 丢掉 |
|---|---:|---:|---:|---:|
| complex | 65 / 80 | 226.9091 / 282 | 80.46% | 55.0909 |
| easy | 94 / 103 | 149 / 163 | 91.41% | 14 |
| medium | 181 / 217 | 537 / 642 | 83.64% | 105 |

medium 题数最多，丢掉 105 分，占失分的 60.3%。complex 得分率最低，丢掉 55.0909 分。easy 只剩 14 分。

Elasticsearch 丢掉 1.0909 分。simple 丢掉 13.0909 分，减去这 1.0909 后剩 12 分；complex 丢掉 55.0909 分，减去这 1.0909 后剩 54 分。其余题型、难度、MySQL、SQLite 的失分都是整数或 .5。所以这道部分分题是 **Elasticsearch × simple × complex**。它的满分记为 M 时，M × (1 − 系数) = 1.0909。其余 59 道未全对题的系数都是 0。

## 提交文件和官方分桶

`result.json` 共 400 条。388 条是 `{question_id, db, query}`，12 条是 `{question_id, db1, query1, db2, query2}`。`db` 计数与官方数据库分项逐项相同：

| 提交字段 | 题数 |
|---|---:|
| `db = SQLite` | 200 |
| `db = MYSQL` | 100 |
| `db = Elasticsearch` | 50 |
| `db = POSTGRESQL` | 38 |
| `db1/db2` 联邦 | 12 |

联邦方向是 MySQL → PostgreSQL 9 题、PostgreSQL → MySQL 3 题，桥都是 `mmsi IN (result)`。

官方若按标注分桶，则这版路由与标注一致。官方若按提交字段分桶，这个一致只说明统计口径。两种读法下，174 分都出在引擎已经落入 MySQL 或 SQLite 桶之后的结果集不等价，以及一道 Elasticsearch 的 ID 或顺序未对齐。

SQLite 200 题按查询里的库名：

| 库 | 题数 | 库 | 题数 |
|---|---:|---|---:|
| soccer_2016 | 29 | professional_basketball | 7 |
| public_review_platform | 21 | shakespeare | 6 |
| books | 17 | simpson_episodes | 6 |
| video_games | 14 | language_corpus | 6 |
| movie_3 | 13 | book_publishing_company | 5 |
| authors | 12 | mental_health_survey | 5 |
| image_and_language | 10 | disney | 5 |
| hockey | 10 | music_platform_2 | 5 |
| olympics | 9 | cookbook | 5 |
| beer_factory | 8 | codebase_comments | 3 |
| | | bike_share_1 | 3 |
| | | genes | 1 |

23 个 BIRD 库里，这版测试提交没有 `citeseer`。官方没有按库回传对错，上表只说明题量，不说明哪一库错了 47 题。

MySQL 100 题按查询里出现的业务库：energy 26，transportation 25，finance_tax 17，ecommerce 10，education_2024 与 education_2025 同时出现 6、只出现其中一个 4，另外 12 题同时用了 `common`。

## 对后续提交的用法

本地执行准确率继续按结果集等价来做，PostgreSQL 和联邦也按等价来做。报官方得分时另开三列：MySQL/SQLite 用等价系数，PostgreSQL/联邦用非空系数，Elasticsearch 用 ID 集合 0.5 + 顺序 0.5。把本地等价准确率直接写成官方得分率，会把 PostgreSQL 和联邦估高。

抢分顺序按丢掉的分，不按全对率：

1. SQLite 的 knowledge 题。47 道 SQLite 系数为 0，合计 144.5 分。knowledge 未全对 48 题，合计 152 分。两者高度重叠。字段说明按 `库.表` 整表注入，业务口径只加写查询要用的那几条。相似案例从训练集里选同一 SQLite 库、同一连接方式的题。
2. MySQL 的 12 道结果集不等价，合计 28.5 分。先核对 `库.表` 前缀、`common` 字典列、教育库跨年是一条 SQL 还是联邦。
3. geo 的 2 道系数为 0，合计 9 分。核对经纬度顺序和距离单位。
4. simple 里 9 道系数为 0，合计 12 分。
5. 那道 Elasticsearch × simple × complex。先把返回 ID 集合对齐，再对顺序。最多 1.0909 分。

PostgreSQL 和联邦这版官方分已经是 59 + 57。查询继续返回非空，官方分就不会从这里掉下去。语义是否与标注等价，用本地执行评测看，不看这两行的 100%。

算术上沿：只把 47 道 SQLite 补到系数 1，总分到 1057.4091 / 1087，得分率 97.28%。只把 SQLite 得分率拉到本版 MySQL 的 89.22%（569.5 × 236 / 264.5 ≈ 508.14），大约多 83 分，总分约 996 / 1087，得分率约 91.6%。knowledge 的 48 道全部补到系数 1，多 152 分，总分 1064.9091 / 1087，得分率 97.97%。这些是同一套判定系数下的上限，默认其余题的系数不变。

下一版官方回传按同一张三张表填写。要看的差是 SQLite 丢掉的分和 knowledge 丢掉的分。PostgreSQL、联邦、calculation、intermediate 这版已经是满分，这四行下降才需要回头查回归。

提交前用脚本核对 `db` 计数仍是 SQLite 200、MySQL 100、Elasticsearch 50、PostgreSQL 38、联邦 12。本地 `test-full-v1` 的调用计数是 MySQL 111、PostgreSQL 52、Elasticsearch 50、SQLite 199（412 次调用，联邦被拆进两次单库调用）。那个计数和本版官方分桶不同，组装提交文件时按题而不是按调用归桶。
