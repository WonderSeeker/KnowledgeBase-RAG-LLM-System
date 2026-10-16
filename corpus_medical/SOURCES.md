# 医疗评测语料来源

抓取时间：2026-10-16（重新抓取用 `scripts/build_corpus_medical.py`）

| 文件 | 字数 | 条目 | 来源 |
|---|---|---|---|
| `01_hypertension.txt` | 10782 | 高血压 | https://zh.wikipedia.org/wiki/%E9%AB%98%E8%A1%80%E5%8E%8B |
| `02_type2_diabetes.txt` | 11385 | 2型糖尿病 | https://zh.wikipedia.org/wiki/2%E5%9E%8B%E7%B3%96%E5%B0%BF%E7%97%85 |
| `03_pneumonia.txt` | 11680 | 肺炎 | https://zh.wikipedia.org/wiki/%E8%82%BA%E7%82%8E |
| `04_anemia.txt` | 4654 | 贫血 | https://zh.wikipedia.org/wiki/%E8%B4%AB%E8%A1%80 |
| `05_asthma.txt` | 11515 | 支气管哮喘 | https://zh.wikipedia.org/wiki/%E6%94%AF%E6%B0%94%E7%AE%A1%E5%93%AE%E5%96%98 |
| `06_depression.txt` | 20247 | 重性抑郁障碍 | https://zh.wikipedia.org/wiki/%E9%87%8D%E6%80%A7%E6%8A%91%E9%83%81%E9%9A%9C%E7%A2%8D |
| `07_coronary_heart_disease.txt` | 2039 | 冠心病 | https://zh.wikipedia.org/wiki/%E5%86%A0%E5%BF%83%E7%97%85 |
| `08_stroke.txt` | 16089 | 脑卒中 | https://zh.wikipedia.org/wiki/%E8%84%91%E5%8D%92%E4%B8%AD |
| `09_osteoporosis.txt` | 2850 | 骨质疏松症 | https://zh.wikipedia.org/wiki/%E9%AA%A8%E8%B4%A8%E7%96%8F%E6%9D%BE%E7%97%87 |
| `10_hypothyroidism.txt` | 7505 | 甲状腺功能减退 | https://zh.wikipedia.org/wiki/%E7%94%B2%E7%8A%B6%E8%85%BA%E5%8A%9F%E8%83%BD%E5%87%8F%E9%80%80 |
| `11_gastritis.txt` | 1393 | 胃炎 | https://zh.wikipedia.org/wiki/%E8%83%83%E7%82%8E |
| `12_urticaria.txt` | 8289 | 荨麻疹 | https://zh.wikipedia.org/wiki/%E8%8D%A8%E9%BA%BB%E7%96%B9 |
| `13_influenza.txt` | 14851 | 流行性感冒 | https://zh.wikipedia.org/wiki/%E6%B5%81%E8%A1%8C%E6%80%A7%E6%84%9F%E5%86%92 |
| `14_tuberculosis.txt` | 10920 | 肺结核 | https://zh.wikipedia.org/wiki/%E8%82%BA%E7%BB%93%E6%A0%B8 |
| `15_hepatitis_b.txt` | 8846 | 乙型肝炎 | https://zh.wikipedia.org/wiki/%E4%B9%99%E5%9E%8B%E8%82%9D%E7%82%8E |
| `16_chronic_kidney_disease.txt` | 3697 | 慢性肾脏病 | https://zh.wikipedia.org/wiki/%E6%85%A2%E6%80%A7%E8%82%BE%E8%84%8F%E7%97%85 |
| `17_gout.txt` | 7927 | 痛风 | https://zh.wikipedia.org/wiki/%E7%97%9B%E9%A3%8E |
| `18_rheumatoid_arthritis.txt` | 6307 | 类风湿性关节炎 | https://zh.wikipedia.org/wiki/%E7%B1%BB%E9%A3%8E%E6%B9%BF%E6%80%A7%E5%85%B3%E8%8A%82%E7%82%8E |
| `19_migraine.txt` | 8878 | 偏头痛 | https://zh.wikipedia.org/wiki/%E5%81%8F%E5%A4%B4%E7%97%9B |
| `20_epilepsy.txt` | 5488 | 癫痫 | https://zh.wikipedia.org/wiki/%E7%99%AB%E7%97%AB |
| `21_alzheimer.txt` | 19614 | 阿尔茨海默病 | https://zh.wikipedia.org/wiki/%E9%98%BF%E5%B0%94%E8%8C%A8%E6%B5%B7%E9%BB%98%E7%97%85 |
| `22_anxiety_disorder.txt` | 4728 | 焦虑症 | https://zh.wikipedia.org/wiki/%E7%84%A6%E8%99%91%E7%97%87 |
| `23_hyperlipidemia.txt` | 6614 | 高脂血症 | https://zh.wikipedia.org/wiki/%E9%AB%98%E8%84%82%E8%A1%80%E7%97%87 |
| `24_heart_failure.txt` | 8728 | 心力衰竭 | https://zh.wikipedia.org/wiki/%E5%BF%83%E5%8A%9B%E8%A1%B0%E7%AB%AD |

## 许可与用途

- 内容来自**中文维基百科**（zh.wikipedia.org），适用 **知识共享 署名-相同方式共享 4.0（CC BY-SA 4.0）**。
- 本仓库**仅将这些条目用作检索评测语料**；版权归各条目贡献者所有，署名与许可信息见上表链接。
- CC BY-SA 要求：使用时署名、并以相同协议共享 —— 若本仓库对外发布，请保留本说明与许可声明。

## 为什么单独一份语料

`corpus/`（Python 文档）上已跑过 3.8 / 4.10~4.12 的结论。换语料等于换评测前提，
**两套结论不可相互引用**；因此医疗语料独立成目录、独立向量库集合、独立评测集。

## 局限（引用前必读）

- 这是**公开百科条目**，不是临床指南/病历，**不能代表真实医疗文档分布**；
  真做医疗 RAG 还需脱敏、术语/缩写表、以及「答案可溯源到原文哪一句」的判分口径。
- 条目内容会随时间被编辑；本目录的 txt 是抓取时点的快照（时间见上）。
