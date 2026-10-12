# A/U 七子图实验说明

## 1. 当前 baseline 的表示与口径

本项目的二分类标签由目录决定：`benign=0`，`mal=1`（恶意）。原预测器不是带单一分类头的判别模型：`infer_binary.py` 先用 VQ-VAE 将图像量化为多尺度 token，再分别以良性和恶意作为条件执行 VAR，累加真实 token 的对数似然，选分数更大的类别。A/U 报告不会更改这个预测规则，也不会参与训练。

本诊断提取**两个条件下真实 VAR 前向的最终 token 预测头输入**。对于每张图和每个类别条件，从 `var_model.head` 的 forward pre-hook 得到模型原有条件化/归一化后的 `[L,C]` token 隐状态，沿 token 维求均值，按 `[benign_C, mal_C]` 顺序拼接为一行 `[2C]`。两次前向都使用完整的原有多尺度 token 路径；hook 在结束后移除。它使用了训练后的 VAR 权重，是明确标注的**诊断性样本表示**，不是原分类器内建的单向量 readout；token 均值也不能代替对数似然分数。没有使用真实标签选择条件，没有把预测概率、logits、类别或二维投影加入特征。对于不同架构，表示位置和维度仍可能使 A/U 缺乏直接可比性。

模型按 `infer_binary.load_models` 的约定恢复：VQ-VAE 权重，以及直接 `state_dict`、`var_wo_ddp` 或 `trainer.var_wo_ddp` 形式的二分类 VAR checkpoint，均严格加载。[上游 VAR 源码](https://github.com/FoundationVision/VAR/blob/main/models/var.py)中 `get_logits()` 明确通过 `head_nm` 后调用 `head`，对应这里的 hook 位置。脚本不自动下载权重。预处理复用 `datasets.build_dataset(binary)`：LANCZOS 缩放到 `round(1.125 × final_reso)`、中心裁剪至 256（d36 为 512）、`ToTensor`、像素映射到 `[-1,1]`。默认 `split=val`。若原评估使用部分尺度或增强邻域，本报告的完整尺度表示并不等同于那些设置下的预测中间量；需在论文中明确报告此诊断定义。

## 2. 运行

在 Linux/Python 3.10 环境中，先按 [REPRODUCE_BINARY.md](REPRODUCE_BINARY.md) 安装依赖并签出 `VAR` 子模块。`requirements.txt` 已增加 SciPy。准备训练后的**二分类** VAR checkpoint、对应 VQ-VAE 权重和 `val/benign`、`val/mal` 图片目录。推荐所有 baseline 使用同一份公共 CSV 清单：

```csv
sample_id,label
val/benign/example_a.png,0
val/mal/example_b.png,1
```

`sample_id` 可以是数据根目录相对路径（如上），也可省略 `val/` 前缀；须唯一且能精确映射到当前 split。CSV 行顺序即导出行顺序。缺失、重复、同一图片使用两个别名、标签冲突均报错。有清单时不抽样或截断。

```bash
python visualize_alignment_uniformity.py \
  --checkpoint VAR/local_output/ar-ckpt-best.pth \
  --vae_ckpt vae_ch160v4096z32.pth \
  --data_path /data/binary_images --split val --depth 16 \
  --sample_manifest /data/common_au_samples.csv \
  --batch_size 1 --device cuda \
  --output_dir results/au/var_d16_best_val \
  --align_alpha 2 --uniform_t 2
```

若尚无公共清单，可先在固定的 `val` split 上无放回抽取最多 1024 张图片并保存清单；请把生成的 `sample_manifest.csv` 作为其他 baseline 的共同输入。默认不按类别重采样。

```bash
python visualize_alignment_uniformity.py \
  --checkpoint VAR/local_output/ar-ckpt-best.pth \
  --vae_ckpt vae_ch160v4096z32.pth \
  --data_path /data/binary_images --split val --depth 16 \
  --max_samples 1024 --seed 42 --batch_size 1 \
  --output_dir results/au/var_d16_best_val
```

仅重绘已导出的全部缓存行，不加载 PyTorch 或 GPU，也不再次抽样：

```bash
python visualize_alignment_uniformity.py \
  --features_npz results/au/var_d16_best_val/features.npz \
  --output_dir results/au/var_d16_best_val_redraw \
  --align_alpha 2 --uniform_t 2
```

`--batch_size` 是图片数；每批对两类分别执行一次 VAR 前向，显存不足时用 1。`--device` 可设 `auto`、`cpu` 或 `cuda`；CPU 可运行，但加载和执行大型 VAR 会很慢。不同 checkpoint、split、层定义及参数应使用独立输出目录。公共清单若很大，成对指标的计算和直方图需要约 `O(N²D)` 时间；先确定可承受的共同诊断子集。

## 3. 输出和计算定义

输出目录包含：

| 文件 | 内容 |
| --- | --- |
| `representation.png`、`representation.svg` | 15×5 英寸七子图；PNG 为 180 dpi |
| `features.npz` | 原始 `[N,D]` 特征、`0/1` 标签、逐行 `sample_ids`；无 pickle |
| `metrics.csv`、`metrics.json` | A、U、`exp(U)`、分组 A、计数和退化信息 |
| `projection_basis.npy` | 本报告共用的非中心化 SVD 二维显示基 |
| `sample_manifest.csv` | 实际使用的 ID 和标签 |
| `metadata.json` | 模型与来源、split、抽样种子、特征定义、预处理和绘图参数 |

高维指标在导出的特征行归一化后计算，使用 `float64`。Alignment 是同标签不同样本的距离 `alpha` 次方，先对每个有效 anchor 的同类伙伴平均，再对 anchor 平均；默认 `alpha=2`。Uniformity 是全部无序不同样本对（同类和异类均包括）的 `log mean exp(-t × squared_distance)`，默认 `t=2`；负值正常。少于两个样本的类别其 Alignment 为 `null`，没有有效 anchor 时总 Alignment 为 `null`；少于两个样本时 Uniformity 为 `null`。零向量被计数并提示，不从高维指标中删除。

左图显示原高维归一化特征的同类/异类 L2 距离直方图。蓝色虚线是**同类 pair 的普通 L2 均值**，并非 Alignment。右侧圆图与角度图仅供显示：对全体样本求一次非中心化二维 SVD；所有类别共用这个基。投影范数接近零的样本只从圆图和角度图排除，仍参加高维指标。圆图用固定带宽 0.10 的 Gaussian KDE；角度图用 `κ=8` 的周期 von Mises KDE。每个分组独立归一化显示，自动色阶不用于比较绝对密度或样本量。不同模型各自拟合的二维基没有相同坐标语义。

## 4. 已执行验证与限制

本工作副本于 2026-10-12 执行了 `tests/test_au_visualization.py`：用逐 anchor/逐 pair 独立计算核对 A/U；确认单向量正缩放和整体正交旋转不改变指标；相同非零向量给出 `A=0, U=0`；检查单样本类别、零向量、重复向量、重复 ID、非法标签、非有限值和投影退化处理。合成数据实际生成并目视检查了 PNG/SVG 七子图；缓存 NPZ 重绘的 A/U 与原报告一致。这些均为 **synthetic 验证，不是 baseline 的真实实验结果**。运行测试：

```bash
python -m unittest discover -s tests -v
```

本机工作副本的 `VAR/` 尚未签出，且没有二分类数据、VAR/VQ-VAE 权重；本机 Python 也未安装 PyTorch。因而尚未完成真实 checkpoint 的特征导出与重绘对照，也不能报告真实 A/U 数值。准备以上文件和环境后运行第 2 节命令，首先核对 `metadata.json`、`sample_manifest.csv` 与 `metrics.json`，再用缓存命令复核数值。本地受限 Windows 沙箱中的旧 Python 3.9 对 `pathlib.Path.resolve()` 存在权限问题；测试运行时仅在启动命令中临时将其替换为 `absolute()` 并设置可写的 Matplotlib 缓存目录，项目代码不包含该环境绕过。目标 Linux/Python 3.10 环境无此本机验证前提。

定义参考 Wang & Isola, *Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere* (ICML 2020)。这里的同标签正样本与按 anchor 平均是监督诊断适配；A/U 几何量不替代分类性能指标。
