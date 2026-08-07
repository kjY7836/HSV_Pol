# 3DMPG affinity 推理

本目录是一份可独立于 `full_library_screening` 外部工程文件运行的 Smina pose 亲和力推理包。
模型代码、权重、8V1Q 受体 PDB/PDBQT 和 5 个测试分子均在本目录中；运行时只需要已安装的
Python 科学计算依赖和 Smina 可执行程序。

## 输入和预测语义

`predict_affinity.py`读取带三维坐标与正确键级的多记录 SDF。对每个 pose，它从受体 PDB
选出距配体重原子 10 Å 内所触及的完整残基，构造与 PDBbind `*_pocket.pdb`一致的局部
pocket，再用 checkpoint 中保存的模型结构、标签均值和标签标准差输出
`predicted_paffinity`。不要把 PDBQT 直接交给亲和力模型；PDBQT 只供 Smina 使用。

模型训练标签混合了 Ki、Kd 与 IC50，因此 `predicted_paffinity`是训练标签尺度上的 pK-like
回归值。输出中的 `approx_concentration_nM = 10^(9-pAffinity)`仅帮助理解数量级，不应当作
严格的 Kd 或实验 IC50。当前 8V1Q pocket（含 DNA/Mg²⁺）相对 PDBbind 训练分布存在域偏移，
结果适合排序和复筛，不等同于实验测定。

## 对正式筛选产物预测

在 `mpgforamp` 环境中进入本目录：

```bash
cd /mnt/sda/ykj/HSV_Pol/full_library_screening/affinity
/home/yangkaijun/miniconda3/envs/mpgforamp/bin/python predict_affinity.py \
  --affinity-input ../runs/<run>/10_affinity_input \
  --output-csv ../runs/<run>/10_affinity_input/model_pose_predictions.csv \
  --summary-json ../runs/<run>/10_affinity_input/local_affinity_summary.json \
  --install-filled-template \
  --resume \
  --device auto
```

`--install-filled-template`会在全部 pose 推理和追踪校验成功后，原子更新正式
`affinity_predictions.csv`，并保留一次性的
`affinity_predictions.before_local_affinity.csv`备份。中途失败不会破坏正式表。脚本同时校验
1-based `ligand_record_index`全集和`pose_id`，避免错配。SDF 按 batch 流式构图，不会把全量
pose 图一次性留在内存中；GPU 可按显存提高`--batch-size`，CPU 默认使用 1 最稳妥。
`--resume`会逐 batch 追加到`model_pose_predictions.csv.partial`；任务中断后用同一命令重跑，
脚本会校验已有记录是 SDF 的连续前缀，然后从下一条 pose 继续。

仓库根目录也提供正式运行入口：

```bash
AFFINITY_DEVICE=cuda:0 AFFINITY_BATCH_SIZE=4 bash scripts/run_local_affinity.sh
```

也可直接传结构文件：

```bash
python predict_affinity.py \
  --receptor-pdb receptor.pdb \
  --ligand-sdf docked_poses.sdf \
  --output-csv pose_predictions.csv \
  --device cpu
```

## 5 分子端到端测试

下面的命令会从 `examples/example_smiles.csv`确定性生成 3D 初始构象，按正式 8V1Q box
运行 Smina（每个分子最多 9 个 pose），再对全部 docked poses 预测亲和力：

```bash
/home/yangkaijun/miniconda3/envs/mpgforamp/bin/python run_examples.py \
  --device auto --cpu 1 --exhaustiveness 12 --num-modes 9
```

结果写入 `examples/results/`：

- `five_examples_smina.sdf`：Smina 三维 pose；
- `pose_affinity_predictions.csv`：逐 pose 的 Smina 分数和 pAffinity；
- `best_per_ligand.csv`：每个示例按最高预测 pAffinity 选出的 pose，并同时报告最低 Smina 分数；
- `run_summary.json`：命令、模型元数据和运行信息。

如 `smina`不在当前环境的 `bin/`或 `PATH`，使用 `--smina /absolute/path/to/smina`。

轻量回归测试（不重新跑 Smina 或加载 512 MB 权重）：

```bash
python -m unittest discover -s tests -v
```

## 目录自包含性

推理所需项目资产只有当前目录内的相对路径。来源和 checkpoint 校验值见
`SOURCE_PROVENANCE.md`。`requirements.txt`记录本机已验证版本；PyTorch 与 PyG 扩展必须安装
ABI 匹配的构建，通常应直接复用 `mpgforamp`，不要仅按版本文本盲目覆盖现有环境。
