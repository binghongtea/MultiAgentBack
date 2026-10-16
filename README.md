# 简单 Python RAG 知识库

基于 Streamlit、SQLite、TF-IDF 和 DeepSeek 的本地知识库问答项目。
支持上传 TXT、Markdown、PDF，自动解析、重叠分块、持久化保存、检索相关片段，再调用 DeepSeek 回答并展示引用原文。

## 快速运行

推荐 Python 3.11 或更新版本。在项目根目录打开 PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

浏览器打开 <http://localhost:8501>。当前工作区已创建 `.venv` 并安装依赖，可以直接执行最后一行。

## 只配置两项

编辑项目根目录已有的 `.env`：

```dotenv
API_KEY=你的DeepSeek密钥
MODEL_NAME=deepseek-flash
```

`MODEL_NAME` 默认采用 [DeepSeek 官方文档](https://api-docs.deepseek.com/) 中的 `deepseek-flash`，也可以填账户可用的其他模型名称。接口地址已固定为
`https://api.deepseek.com/chat/completions`，无需另外配置地址。环境变量中的同名配置优先于 `.env`。
修改 `.env` 后重新提问即可，无需重启。
从仓库重新获取项目时，可先将 `.env.example` 复制为 `.env`。

## 使用方法

1. 在左侧上传文件并点击「导入知识库」，或点击「导入示例文档」。
2. 输入问题，例如「员工年假有多少天？」「费用报销需要哪些材料？」。
3. 回答中的 `[1]`、`[2]` 对应「查看引用原文」中的片段；PDF 还会展示页码。
4. 同名文件的新内容会替换旧内容，相同文件重复导入不会增加副本。可以在「管理文档」中删除文档。

也可以直接问「标题给我」「这份文档的标题是什么？」「文档列表」。这类问题查询本地文档信息，
Markdown 优先显示一级标题，其他文件显示文件名；不依赖正文关键词匹配，也不需要调用模型。

未配置 Key 或接口调用失败时，页面明确提示原因，并展示本地检索原文；这不是模型生成的回答。
没有相关片段时不会调用 API。对话显示历史记录，但每次问题独立检索和回答，暂不支持依赖上下文的追问。

## 工作流程

```text
TXT / Markdown / PDF
        ↓ 文本解析（PDF 保留页码）
重叠分块（每块最多 600 字符，重叠 100 字符）
        ↓
SQLite 本地持久化
        ↓ 用户提问
字符 TF-IDF 向量 → 余弦相似度 → Top-K 相关片段
        ↓
问题 + 片段 + 引用编号 → DeepSeek → 回答及原文来源
```

这是轻量入门实现：检索使用中文字符 2～4 gram 的 TF-IDF 稀疏向量，依赖字面相似性，**不是语义 Embedding 检索**。
不需要下载模型或部署向量数据库。索引在每次提问时基于当前文档重建，适合小型知识库；同义表达和大规模文档检索后续可升级为 Embedding + 向量数据库。
相似度只是检索排序分数，不是回答的可信概率。模型引用由提示词约束，重要结论仍应对照原文核实。

## 文件结构

```text
app.py                  # Streamlit 页面：上传、文档管理、问答、引用展示
rag/
  documents.py          # TXT / Markdown / PDF 解析与重叠分块
  engine.py             # SQLite 保存、TF-IDF 检索、RAG 问答流程
  llm.py                # 读取两项配置、调用 DeepSeek、处理接口错误
examples/员工手册.md    # 可一键导入的虚构示例
tests/                  # 核心流程和界面测试（不调用真实 API）
.env                    # 仅 API_KEY、MODEL_NAME，已加入 .gitignore
.env.example            # 配置模板
requirements.txt        # 项目依赖
data/knowledge.db       # 运行时自动创建，保存文档片段
```

## 验证

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

测试覆盖中文检索、重启后的数据恢复、同名文档替换、删除、PDF 页码、API 请求格式与失败降级、界面导入及问答流程。API 测试使用模拟响应，不消耗账户额度。

## 使用范围

- 单机学习项目，默认仅监听 `127.0.0.1`，没有用户登录或多租户隔离。
- 上传文件最多 20 MB；PDF 仅提取文本，扫描件需要先做 OCR。
- 文档解析和检索在本地完成；提问及检索命中的片段会发送给 DeepSeek。
- SQLite 中保存的是解析后的片段，不保存原始上传文件。知识库会跨重启保留，对话历史只保留在当前页面会话。

接口与库参考：[DeepSeek API](https://api-docs.deepseek.com/)、[Streamlit](https://docs.streamlit.io/)、[TF-IDF](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)。
