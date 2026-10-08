# 评测语料来源

抓取时间：2026-10-08（重新抓取用 `scripts/build_corpus.py`）

| 文件 | 字数 | 来源 |
|---|---|---|
| `01_python_intro.txt` | 8487 | https://docs.python.org/zh-cn/3/tutorial/introduction.html |
| `02_controlflow.txt` | 17830 | https://docs.python.org/zh-cn/3/tutorial/controlflow.html |
| `03_datastructures.txt` | 11962 | https://docs.python.org/zh-cn/3/tutorial/datastructures.html |
| `04_modules.txt` | 11720 | https://docs.python.org/zh-cn/3/tutorial/modules.html |
| `05_inputoutput.txt` | 9326 | https://docs.python.org/zh-cn/3/tutorial/inputoutput.html |
| `06_errors.txt` | 12271 | https://docs.python.org/zh-cn/3/tutorial/errors.html |
| `07_classes.txt` | 14396 | https://docs.python.org/zh-cn/3/tutorial/classes.html |
| `08_stdlib_overview.txt` | 5673 | https://docs.python.org/zh-cn/3/tutorial/stdlib.html |
| `09_json.txt` | 14453 | https://docs.python.org/zh-cn/3/library/json.html |
| `10_sqlite3.txt` | 41959 | https://docs.python.org/zh-cn/3/library/sqlite3.html |
| `11_pathlib.txt` | 34753 | https://docs.python.org/zh-cn/3/library/pathlib.html |
| `12_regex.txt` | 32232 | https://docs.python.org/zh-cn/3/library/re.html |

## 许可与用途

- 内容来自 Python 官方文档的中文翻译（docs.python.org），适用 **Python 软件基金会许可协议（PSF License）**。
- 本仓库**仅将这些文件用作检索评测语料**（占位示例），不作其它用途；版权归 Python Software Foundation 所有。
- 若本仓库需要对外发布，请保留本说明与 PSF 许可声明。

## 为什么选这些文档

1. **题材中性**：纯技术文档，不含任何个人信息（刻意避开本机规划类文档）。
2. **主题互相独立**：12 个不同主题，使「文档级」检索评测（hit@k / MRR）有区分度；
   若只用一篇长文，文档级指标会失去意义。
3. **可复现**：来源 URL 与抓取方式都写在本目录与 `scripts/build_corpus.py` 里。
