# K3 原始笔记：2026-09-09 脱敏快照

[返回项目](../../README.md) · [主题与实验索引](../../evidence-map.md)

## 用途与证据边界

本目录保存本地 `K3/209-note.md` 的 2,230 行历史记录，归档日期 2026-09-10。
原本地文件未修改，也没有重新获取远端原件；日期是快照标签，不表示当前服务状态。
原文的“当前”“已修复”“根因”“通过”等措辞均属于当时记录，可能被后文更正。
**不要把这里当作操作手册，不要直接执行其中的启动、清理或修改命令。**

档案只保留笔记文本；其中提到的 wheel、CSV、tensor dump 和服务日志没有随之导入，
其 basename 是查找线索，不保证工件仍存在。当前提炼结论以项目入口指向的主题页为准。

## 按需读取

- [01-bootstrap-and-environment.txt](./01-bootstrap-and-environment.txt)：原文 1–338 行。
- [02-conv-dtype-and-pp1.txt](./02-conv-dtype-and-pp1.txt)：原文 339–675 行。
- [03-logprob-and-first-pp-dumps.txt](./03-logprob-and-first-pp-dumps.txt)：原文 676–1238 行。
- [04-hidden-and-attnres-dumps.txt](./04-hidden-and-attnres-dumps.txt)：原文 1239–1810 行。
- [05-same-tp32-pp-comparison.txt](./05-same-tp32-pp-comparison.txt)：原文 1811–1902 行。
- [06-dp2-wheel-and-deepep-hang.txt](./06-dp2-wheel-and-deepep-hang.txt)：原文 1903–2060 行。
- [07-pd-topology-and-pr-review.txt](./07-pd-topology-and-pr-review.txt)：原文 2061–2230 行。

通常先读实验索引，只读取选定片段中的相关行。大段 token/logprob 和 hidden dump 不应
进入默认上下文；精度结论可先看 05，连续请求问题看 06，七机配置看 07。

## 保真与脱敏

- 行序与行数一一对应；未压缩日志、未删去失败实验，也未替原文修正技术结论。
- 用户路径、实际账户、IP/子网、主机简称、私有镜像、容器名和模型部署路径已替换。
  同组已识别主机使用稳定占位符；脱敏后的命令不再可直接执行。
- 每片最后增加一行 `[ARCHIVE_SEGMENT_END]`，不属于原文，用于保留片尾空行。
- 原文行号 = 片段行号 + 起始行号 − 1。例如 05 的第 1 行对应原文 1811。
- [manifest.json](./manifest.json) 记录原文件 SHA-256、各片段范围与脱敏后文件 SHA-256。
  原文件哈希用于本地来源核对，不意味着脱敏副本与原文件字节相同。
- 后续补充另建日期快照；本目录不承担不断更新的“最新结论”职责。
