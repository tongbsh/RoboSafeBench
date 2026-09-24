# RoboSafeBench 项目页

公开宣传页，结构对齐 [LIBERO](https://libero-project.github.io) / [LIBERO-Safety](https://libero-safety.github.io)：标题与链接、基准叙事、L0–L3 demo、指标、引用。

页面是**纯静态文件**，没有构建步骤。Demo 区只播执行视频；论文链接、作者、真实 rollout 按下面填。

## 本地预览

在仓库根目录：

```bash
python -m http.server 8765 --directory website
```

浏览器打开 <http://127.0.0.1:8765>。

## 要改的配置

编辑 [`site.config.js`](site.config.js)：

| 字段 | 用途 |
|------|------|
| `paperUrl` / `arxivUrl` / `codeUrl` / `docsUrl` | 论文、代码、文档。空字符串会显示为「即将放出」 |
| `venue` | 例如 `ICLR 2027`；空则显示「审稿中」 |
| `teaserYoutube` | 全页预告片的 YouTube id |
| `videos` | 可选。本仓**不附带**论文评测视频。缺文件时页面只显示占位；可自行把 mp4 放到 `assets/videos/`（该目录 gitignore），或在这里填外链 / YouTube id。 |

文案在 [`js/i18n.js`](js/i18n.js)，`en` / `zh` 各一份。右上角切换语言，选择会记在 `localStorage`。

匿名审稿期间**不要**把真实作者写进页面。Camera-ready 时改 `index.html` 的作者行，并换掉 BibTeX。

## 发布到 GitHub Pages

本地改完后，在仓库根目录执行：

```bash
bash website/publish.sh
```

脚本会把 `website/` 推到匿名仓 `bshtong-demo/RoboSafeBench`（提交作者是 RoboSafeBench，不会用本机 git 姓名/学校邮箱）。一两分钟后刷新：

https://bshtong-demo.github.io/RoboSafeBench/

部署私钥在本机 `website/.deploy/id_ed25519`（已 gitignore，不要提交）。换机器时把该文件拷过去，或设置 `ROBOSAFE_PAGES_KEY`。

## 和仓库其余部分的关系

- 公开品牌名：**RoboSafeBench**
- 评测代码包名仍是 `dynisafety`（不要写进公开页）
- 协议与指标以 `docs/user/eval.md` 和 `configs/` 为准。本目录只做展示，不附带论文表图或评测 rollout。
