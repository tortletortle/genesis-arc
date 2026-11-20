# Genesis ARC

> Private 开发仓 · V1.9 可运行源码已迁入（原始实验数据及完整 Git 历史仍待同步）

## 研究目标

研究数字生命在受控离散环境中的自主规律发现、技能迁移、有限记忆及连续学习。当前为研究原型，不是通用智能或完整生态种群系统。

## 当前完成度

- V1.9：运行入口 run.py、V1.7 算法依赖 v17_baseline.py、统计脚本 analyze.py、requirements.txt、README 和研究报告已迁入。
- V1.8：研究报告已迁入；其完整源码和原始数据尚待迁入。
- 以上七个文件已逐一核对，Git blob SHA 与本地原件一致。
- 本地原件自测及两组单种子冒烟实验通过（10%/20% 噪声，各72条检查点、24条记忆记录），统计脚本运行成功；尚未执行远程 CI。
- 本地原始 V1.8 / V1.9 提交、标签及历史实验数据保存在 genesis-arc-history.bundle；远程此次采用新提交导入，尚未迁入原始 Git 历史及完整 CSV。不要将其视为完整迁移。

## 运行

```bash
cd versions/v1.9
pip install -r requirements.txt
python -c 'import run; run.selftest()'
python run.py --seeds 8 --rounds 6 --out results
python run.py --seeds 8 --rounds 6 --noise .2 --offset 10000 --out robustness
python analyze.py
```

## 研究报告

- [V1.8](versions/v1.8/V1.8_REPORT.md)
- [V1.9](versions/v1.9/V1.9_REPORT.md)

## 工程约定

1. 一个独立项目对应一个 GitHub 仓库。
2. 开发阶段默认 Private。
3. 保留原始 Git 历史及版本标签；远程已有独立初始化提交，不允许直接强制覆盖。
4. 保留实验数据、测试报告和变更记录；不得将未验收功能标记为完成。
5. 尚未选择开源许可证。
