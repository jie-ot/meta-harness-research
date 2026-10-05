# LawBench Harness RSI 四项预实验 v3

先阅读 `执行任务书.md`。本版本替代旧20题评分、D=0/15/30的预实验设置；四项实验框架保留，采用训练200、评分100、额外反馈池100、审计100，D=0/50/100。

交给执行agent的入口要求：完成必要代码修改、假LLM验收、统一驱动、四项真实实验与最终分析。包中没有未经测试却宣称可直接付费运行的完整驱动器；模型/API启动代码由执行agent按任务书实现。

`prepare_data.py` 为纯标准库数据准备脚本，支持下载固定官方数据，或传入手动下载的源JSON，逐题核对后生成新版数据和manifest；不调用LLM，不执行压缩包中的代码，不覆盖原文件。

```powershell
python -m unittest -v test_prepare_data.py
python .\prepare_data.py --archive "D:\实际路径\text_classificition.zip" --out ".\prepared_data_v3"
```

单元测试使用合成数据，只验证程序逻辑，不代表真实500题已全量匹配。当前沙箱未成功下载完整源文件；真实匹配由执行agent在联网环境完成，失败时不得启动付费实验。

需要同时提供原始 `text_classificition.zip` 以及运行机器上有效的模型认证。不要将API密钥提交到代码仓库或结果报告。
