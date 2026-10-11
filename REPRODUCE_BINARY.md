# 良性 / 恶意二分类复现说明

本文从仓库根目录执行命令，目标环境为 Linux、Python 3.10、NVIDIA CUDA 12.4。

## 1. 方法与适配范围

本项目的 `eval.py` 使用 VQ-VAE 把图片变成多尺度 token，再对每个候选类别计算 VAR 的条件对数似然，选择分数最高的类别。本适配保留这套分类计算。二分类候选类别固定为 `benign=0` 和 `mal=1`，没有恶意软件家族标签，也不使用 ImageNet synset 子集。

原 A-VARC 仓库没有独立的二分类训练脚本；其训练依赖 `VAR` 子模块。`train_binary.py` 只把上游训练脚本的 ImageNet 数据构建器替换为两类目录读取器。模型、损失、优化器、学习率调度、分布式采样、验证、检查点和自动续训仍由上游 `VAR/train.py` 完成。原论文的 A-VARC+ 还包含 CCA 微调，但此仓库没有 CCA 训练实现；这里复现的是 VAR 条件似然分类 baseline，不能把结果标成 A-VARC+。[论文](https://arxiv.org/html/2510.12060)、[上游 VAR 训练说明](https://github.com/FoundationVision/VAR#training-scripts)。

## 2. 准备代码和 Python 环境

当前工作副本的 `VAR/` 是尚未签出的 Git 子模块。将仓库放到 Linux 服务器后，在仓库根目录运行：

```bash
git submodule update --init --recursive
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
```

`requirements.txt` 锁定 PyTorch 2.5.1 / torchvision 0.20.1 的 CUDA 12.4 wheel；该组合见 [PyTorch 官方历史版本说明](https://pytorch.org/get-started/previous-versions/)。不要再执行旧 README 中的 conda / cu126 安装命令。`flash-attn` 和 `xformers` 不是运行所必需的，未列入依赖。执行上面命令之前，需要服务器的 NVIDIA 驱动可运行 CUDA 12.4 程序；训练需要可用 GPU。

## 3. 准备数据与 VQ-VAE 权重

`--data_path` 指向包含各划分目录的根目录，例如 `/data/binary_images`：

```text
/data/binary_images/
├── train/
│   ├── benign/  (图片文件)
│   └── mal/     (图片文件)
└── val/
    ├── benign/  (图片文件)
    └── mal/     (图片文件)
```

每个划分下必须正好有 `benign` 与 `mal` 两个非空类别目录，目录内为 torchvision 支持的图片格式。目录名决定标签：`benign=0`、`mal=1`；`train` 与 `val` 必须使用相同命名。若以后划分改名，训练时传 `--train_split` / `--val_split`，评估时传 `--split`，无需改代码。

上游 VAR 训练需要预训练 VQ-VAE 权重 `vae_ch160v4096z32.pth`。从 [VAR 官方权重页面](https://huggingface.co/FoundationVision/var/blob/main/vae_ch160v4096z32.pth) 获取后，放在**仓库根目录**；这是上游训练脚本读取的相对路径。训练脚本在该文件不存在时会尝试自动下载，但建议提前放好以便明确权重来源。ImageNet 的 `var_d16.pth` 是 **1000 类** 模型，不能直接当作本二分类检查点。

## 4. 训练

在仓库根目录运行。下面的 d16、256×256、全局 batch 768、200 epochs、FP16、`alng=1e-3`、`wpe=0.1` 与 [上游 VAR d16 训练命令](https://github.com/FoundationVision/VAR#training-scripts)一致：

```bash
torchrun --standalone --nproc_per_node=8 train_binary.py \
  --data_path=/data/binary_images \
  --train_split=train --val_split=val \
  --depth=16 --bs=768 --ep=200 --fp16=1 --alng=1e-3 --wpe=0.1
```

上游其他默认值继续生效：AdamW、基础学习率 `1e-4`（按 batch 缩放）、权重衰减 `0.05`、梯度裁剪 `2`、标签平滑 `0`、默认不水平翻转、约 `1.125×` 缩放后训练随机裁剪和验证中心裁剪、像素映射到 `[-1,1]`。上游默认每 10 个 epoch 和最后一个 epoch 验证并保存检查点；目录为 `VAR/local_output/`，包括 `ar-ckpt-last.pth` 与 `ar-ckpt-best.pth`。重复同一命令会按上游逻辑自动续训。其他上游参数仍可直接传给 `train_binary.py`。

上游命令按 8 GPU 设计。如果服务器 GPU 较少，可通过上游 `--ac` 做梯度累积，例如 1 GPU 使用 `--bs=768 --ac=8` 时单次前向 batch 约为 96、有效 batch 约为 768。若显存不足，继续增大 `--ac`。这是资源调整，不是论文报告的 8 GPU 配置；建议记录实际 GPU 数和 `--ac`。

## 5. 验证或评估任意有标签划分

训练完成后，用二分类检查点运行原有的条件似然分类器：

```bash
python eval.py \
  --dataset binary --data_path /data/binary_images --split val \
  --depth 16 --batch_size 1 \
  --model_ckpt VAR/local_output/ar-ckpt-best.pth \
  --vae_ckpt vae_ch160v4096z32.pth \
  --num_candidate_list 1 --num_sample_list 1 --num_scale_list 10
```

若推理划分叫 `zero_test`，只需把 `--split val` 改为 `--split zero_test`，并准备 `zero_test/benign` 与 `zero_test/mal`。`--data_path` 始终指向这些划分的共同父目录。评估会输出 Top-1 准确率；可加 `--save_json` 保存每张图片的预测与分阶段分数。结果默认位于 `outputs_formal/binary_<split>/log_likelihood/`。设置 `--extra ''` 可直接写入 `outputs/`，或用 `--output_dir` 指定根目录。

原脚本默认一次用 10 个尺度、不加平滑。若要启用项目已有的似然平滑，可把评估命令改为 `--num_sample_list 10 --sigma 0.1`，计算成本会增加。论文在 ImageNet 多类任务上的候选剪枝参数不能照搬到只有 2 个类别的任务；单阶段同时比较两类即可。[论文中的平滑与剪枝说明](https://arxiv.org/html/2510.12060)。

可选的原有热力图入口也接受相同二分类数据与检查点：

```bash
python vis.py --dataset binary --data_path /data/binary_images --split val \
  --depth 16 --batch_size 1 \
  --model_ckpt VAR/local_output/ar-ckpt-best.pth \
  --vae_ckpt vae_ch160v4096z32.pth
```

## 6. 结果解释与检查

- 训练阶段的 `val` 是生成模型的 token 损失 / token 准确率验证；**分类准确率**请运行 `eval.py` 查看。
- 本适配用二分类 VAR 从头训练类别条件层，沿用官方预训练 VQ-VAE；不支持把 1000 类 VAR 权重直接改标签加载。
- 对比实验请固定数据划分、VQ-VAE 权重、训练检查点、随机种子及评估参数。上游训练默认 `seed=None`；需要固定时传 `--seed=<整数>`。评估默认 `--seed=0`。
- 当前工作副本没有签出 `VAR` 子模块，也没有数据和权重；因此只能做静态代码验收，完整训练 / GPU 验证需要在服务器准备上述文件后执行。
