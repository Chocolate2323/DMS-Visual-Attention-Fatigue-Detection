# YawDD 数据集评估报告

本次评估使用本地 `data/local/archive/Mirror` 中的 YawDD Mirror 子集。该子集文件名包含 `Normal`、`Talking`、`Yawning`、`TalkingYawning` 标签，因此可以直接用文件名构造视频级真值。

## 评估设置

- 数据目录：`data/local/archive/Mirror`
- 视频数量：319
- 标签分布：
  - `Normal`：105
  - `Talking`：100
  - `Yawning`：101
  - `TalkingYawning`：13
- 正类定义：`Yawning` 和 `TalkingYawning`
- 负类定义：`Normal` 和 `Talking`
- 抽帧策略：`frame_stride=10`
- 输出目录：`outputs/yawdd_mirror_eval`

运行命令：

```bash
PYTHONUNBUFFERED=1 MPLCONFIGDIR=/tmp/matplotlib XDG_CACHE_HOME=/tmp \
/home/xuteng/miniconda3/envs/machine_learning/bin/python scripts/evaluate_yawdd.py \
  --input-dir data/local/archive/Mirror \
  --output-dir outputs/yawdd_mirror_eval \
  --frame-stride 10
```

输出文件：

```text
outputs/yawdd_mirror_eval/summary.csv
outputs/yawdd_mirror_eval/summary.json
outputs/yawdd_mirror_eval/metrics.json
```

## 总体结果

### 当前事件规则

当前事件规则是系统内部的 `yawn_count > 0`，也就是嘴巴张开超过配置阈值并持续足够时间后，才记为一次打哈欠。

| 指标 | 数值 |
|---|---:|
| Accuracy | 0.7931 |
| Precision | 1.0000 |
| Recall | 0.4211 |
| F1 | 0.5926 |
| TP / FP / TN / FN | 48 / 0 / 205 / 66 |

结论：当前事件规则非常保守，几乎没有误报，但漏检较多。

### 默认 MAR=0.60 规则

该规则以视频内最大 `MAR >= 0.60` 判断该视频存在打哈欠。

| 指标 | 数值 |
|---|---:|
| Accuracy | 0.8150 |
| Precision | 0.9825 |
| Recall | 0.4912 |
| F1 | 0.6550 |
| TP / FP / TN / FN | 56 / 1 / 204 / 58 |

结论：默认 MAR 阈值仍然偏高，精确率很好，但召回率只有约 49%。

### 本次扫描得到的最佳 MAR 阈值

在本次评估结果上扫描 `MAR` 阈值，最佳点为：

| 阈值 | Accuracy | Precision | Recall | F1 |
|---:|---:|---:|---:|---:|
| 0.36 | 0.8997 | 0.9020 | 0.8070 | 0.8519 |

对应混淆矩阵：

```text
TP = 92
FP = 10
TN = 195
FN = 22
```

结论：对 YawDD Mirror 子集而言，`MAR=0.36` 比当前默认 `0.60` 更合适。

## 分类别表现

### 按标签观察

| 标签 | 数量 | MAR=0.60 预测正类数 | MAR=0.36 预测正类数 | 平均 max_mar |
|---|---:|---:|---:|---:|
| Normal | 105 | 0 | 2 | 0.099 |
| Talking | 100 | 1 | 8 | 0.230 |
| Yawning | 101 | 52 | 84 | 0.579 |
| TalkingYawning | 13 | 4 | 8 | 0.452 |

观察：

- `Talking` 的嘴部动作会抬高 MAR，是降低阈值后的主要误报来源。
- `TalkingYawning` 比纯 `Yawning` 更难，因为说话和哈欠混在一起，嘴部开合持续性不稳定。

### 按性别观察

| 分组 | MAR=0.60 F1 | MAR=0.36 F1 |
|---|---:|---:|
| Female | 0.602 | 0.796 |
| Male | 0.705 | 0.907 |

观察：男性样本在本次规则下更容易检测，女性样本漏检更多，可能与嘴部张开幅度、脸部角度、光照和遮挡有关。

### 按眼镜类型观察

| 分组 | MAR=0.60 F1 | MAR=0.36 F1 |
|---|---:|---:|
| Glasses | 0.800 | 0.909 |
| NoGlasses | 0.571 | 0.813 |
| SunGlasses | 0.600 | 0.889 |

观察：`NoGlasses` 组在默认阈值下召回最低，降低阈值后改善明显。墨镜组样本较少，结果只能作为参考。

## 误判分析

### 默认 MAR=0.60 的问题

- 误报极少，只有 1 个 `Talking` 被判为打哈欠。
- 漏检较多，很多 `Yawning` 视频的最大 MAR 在 `0.36 - 0.60` 之间。
- 说明当前阈值更适合“嘴巴非常大幅张开”的场景，而 YawDD 中不少哈欠幅度较小。

### MAR=0.36 的问题

- 召回率从 0.4912 提升到 0.8070。
- 误报从 1 个增加到 10 个，主要来自 `Talking`，因为说话时嘴巴也可能张开到较大。
- 说明单纯依赖 `max_mar` 会把部分说话动作误认为哈欠。

## 当前项目效果结论

当前系统在 YawDD Mirror 子集上的人脸检测非常稳定：

```text
face_found_ratio 平均值 = 0.9983
face_found_ratio 中位数 = 1.0000
```

但打哈欠检测规则偏保守：

- 如果使用项目默认阈值，适合“少误报”的场景，但漏检明显。
- 如果目标是数据集上的打哈欠识别，建议将 `fatigue.mar_yawn_threshold` 调到约 `0.36`。
- 仅靠 `max_mar` 不足以区分 `Talking` 和 `Yawning`，后续应加入持续时间、MAR 上升/下降形状、嘴部开合稳定性等时间序列特征。

## 建议改进

1. 使用 `configs/yawdd_eval.yaml` 在 YawDD 上测试更低的 MAR 阈值：

```bash
python scripts/evaluate_yawdd.py \
  --input-dir data/local/archive/Mirror \
  --output-dir outputs/yawdd_mirror_eval_config036 \
  --config configs/yawdd_eval.yaml \
  --frame-stride 10
```

2. 不要只用 `max_mar` 判断哈欠，建议增加：

- `MAR > threshold` 的持续时间
- 连续高 MAR 帧比例
- MAR 峰值宽度
- 嘴巴张开速度和闭合速度
- `Talking` 和 `Yawning` 的时间序列分类器

3. 对课程项目而言，可以把当前规则作为 baseline，再用 YawDD 的视频级标签训练一个简单分类器：

```text
输入特征：
max_mar, mean_mar, high_mar_ratio, yawn_count, face_found_ratio, max_fatigue_score

模型：
RandomForest / SVM
```

4. YawDD 主要是打哈欠数据集，不是完整疲劳驾驶数据集。因此本次结果更适合评价“打哈欠检测”，不能完全代表 PERCLOS 闭眼疲劳检测效果。
