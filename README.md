# HSV Pol（H9E937）全库虚拟筛选

面向 HSV-1 DNA 聚合酶催化亚基 UL30（UniProt `H9E937`）的可追溯虚拟筛选工程。流程从约
1,000 万条原始化合物记录中形成 20 万个三维预筛母体，随后将固定 10,000 个未标注来源
候选与全部计算可用的活性记录来源送入 `8V1Q` WT 受体 Docking；无法生成经验证 3D 构象的
活性记录会被排除并写入 `activity_3d_failures.csv`。活性记录不被解释为
HSV Pol 阳性，未标注记录也不被解释为阴性。

## 当前方案

```text
原始记录
→ 标准化、去盐、去重、可用性检查和宽松理化过滤
→ 全库 PNU-183792 2D/Pharm2D 打分及分层骨架覆盖
→ 固定 200,000 个 3D 预筛母体
→ PNU 通道 3D 富集及其他通道 3D 可生成性检查
→ 全部计算可用活性记录 + 10,000 个未标注候选
→ 每个母体最多一个 pH 7.4/规范互变异构状态和一个 ETKDGv3 初始构象
→ 排除无法生成经验证 3D 初始构象的母体并输出失败清单
→ 仅 8V1Q-WT 局部 Smina Docking，每个输入状态最多输出 9 个 pose
→ Docking 后 PNU 三维构象相似性
→ 独立 affinity 输入包（受体 PDB、pose SDF、全部非 affinity 分数）
→ 外部 PDBbind 模型填写 predicted_paffinity
→ WT 综合评分
```

每个成功嵌入的母体只进入一个化学状态和一个初始构象。优先采用 Open Babel 在 pH 7.4 生成的结构，
再用 RDKit 选择确定性的规范互变异构体；保留来源已定义的立体化学，不枚举未定义立体中心。
无法生成经验证 3D 初始构象的母体不进入 Docking，并记录在
`06_ligand_states/ligand_state_failures.csv`。这是一种可复现的筛选状态，不代表完整微观
pKa/微观状态分布。

## 结构与 Docking

| 结构 | 用途 | Docking 受体 |
|---|---|---:|
| 8V1Q | H9E937 开放态，保留 UL30、DNA、primer 末端 DOC 和 Mg²⁺ | 是 |
| 8V1R | 将实验 dTTP 位点的 box 转移到 8V1Q | 否 |
| 7LUF–PNU | PNU-183792 实验构象及前/后 Docking 相似性参考 | 否 |

唯一正式协议：

```text
protocol       = 8v1q_open_wt
box center     = (147.660, 145.083, 124.388) Å
box size       = 24 × 24 × 28 Å
exhaustiveness = 12
num_modes      = 9
```

8V1Q 没有共晶小分子，因此不伪造 redocking。Mg²⁺可能使金属配位基团获得偏乐观分数，相关
pose 需要在最终精细复筛中单独检查。

## 综合评分

外部模型提供的 `predicted_paffinity` 先转换为：

```text
A = sigmoid((predicted_paffinity - 6.0) / 1.0)
```

对每个母体选择 affinity 证据最好的 pose，最终分数为：

```text
Final = 100 × (
    0.55 × A
  + 0.20 × postdock_3d_similarity
  + 0.15 × docking_percentile
  + 0.10 × QED
)
```

最终公式不包含 ligand efficiency、pose quality、soft-alert 扣分，也不再使用包含警示惩罚的
复合 chemical-quality 项；化学质量分量仅为原始 RDKit QED。PAINS/Brenk 仍保留在前期审计
字段和部分候选通道排序中，不进入最终扣分公式。

## 工程结构

```text
config/       正式与 10,000 条抽样配置
references/   当前使用的 PDB/mmCIF、受体及实验配体结构
scripts/      只包含单条 Python 命令的 Shell 入口
src/          Python 包源码
tests/        功能与数据契约测试
runs/         运行时生成目录，不纳入 Git
```

原始千万级化合物库位于工程外的 `../data`，体积过大，不纳入本 Git 仓库；版本库包含流程实际
依赖的小型结构参考数据。`references/manifest.json`记录结构身份和校验信息。

## 环境安装

推荐使用现有 `mpgforamp` Conda 环境：

```bash
conda activate mpgforamp
cd /mnt/sda/ykj/HSV_Pol/full_library_screening
```

运行依赖包括 Python 3.10+、RDKit、NumPy、Pandas、SciPy、Open Babel 和 Smina。多节点
协调不依赖`mpi4py`，但提交作业后必须由平台的`module load mpi`提供`mpirun`。当前 Smina
默认位置为：

```text
/home/yangkaijun/miniconda3/envs/mpgforamp/bin/smina
```

## 运行

`scripts/*.sh`不包含环境设置、路径推断或流程逻辑，每个文件只有一条`python -m ...`命令。
在仓库根目录执行：

```bash
# 功能测试
bash scripts/run_tests.sh

# 10,000 条原始记录抽样：完成除外部 affinity 预测和最终评分外的全部步骤
bash scripts/run_sample_10000.sh

# 全量：完成筛选、Smina、Docking 后 3D 和 affinity 包导出
bash scripts/run_full.sh

# 已经准备好 jobs.tsv 时，只运行全量 Smina 和后处理
bash scripts/run_smina.sh

# 使用仓库内置的本地 3DMPG 模型预测并原子回填 predicted_paffinity
AFFINITY_DEVICE=cuda:0 AFFINITY_BATCH_SIZE=4 bash scripts/run_local_affinity.sh

# affinity 预测完成后执行最终评分
bash scripts/run_final_score.sh
```

当前固定使用2个CPU节点、每节点64个物理核，总计128核。`run_full.sh`和`run_smina.sh`仍只有一条
Python命令；Python入口会调用系统`mpirun -np 2 --map-by ppr:1:node:PE=64 --bind-to core`，在每个节点启动一个
协调rank，再由每个rank创建64个本地工作进程。Smina每节点并发64个任务，每个任务`cpu=1`。

Smina 支持逐 chunk 断点续跑。每个成功输出旁会写入`docked_*.sdf.done.json`，其中包含任务输入、
受体、Smina程序和参数的指纹。重提交时会以`sanitize=False`读取旧SDF，校验所有输入state均有
可追踪pose且打分字段有效；通过校验的旧输出直接跳过，升级前生成的无marker输出会自动补建marker。
需要重算的chunk先写唯一临时SDF并完成相同校验，成功后才原子替换正式结果，所以作业失败或被中断
不会覆盖已有输出。若确实需要强制重算单个chunk，应先把对应正式SDF及其`.done.json`移出输出目录。

Docking收集阶段会严格sanitize每个pose。针对已确认的Smina/Open Babel亚砜写出问题，收集器仅将
非法的`S+=[O-]`规范化回输入所用的`S+-[O-]`单键表示，并在汇总SDF属性及
`08_docking_results/summary.json`中记录修复类型、pose数和state数；其他化学错误仍会立即失败。

应通过调度器申请资源，不能在登录节点直接运行全量流程。提交文件示例：

```bash
#!/bin/bash
#JSUB -J hsv-pol-full
#JSUB -q cpu_x86fat
#JSUB -n 128
#JSUB -R "span[ptile=64]"
#JSUB -o output.%J
#JSUB -e error.%J

module load mpi
source /目标路径/envs/mpgforamp/bin/activate
cd /目标路径/HSV_Pol/full_library_screening
bash scripts/run_full.sh
```

程序会验证必须正好出现2个不同主机名，且每个rank能访问至少64个CPU；如果MPI把两个rank放到
同一节点或CPU亲和集不足，会在开始计算前失败。
所有节点必须能以相同路径访问仓库、`../data`、Conda-pack解压环境和`runs/`共享目录。标准化、
2D、20万3D、构象生成、Smina和Docking后3D均分片到2节点；配额选择、去重归并、SDF汇总和
affinity输入包由rank 0执行。任何已调度分子没有产生Smina pose都会直接报错。

## Affinity 模型交付

Smina 和 Docking 后 3D 完成后自动生成：

```text
runs/<run>/10_affinity_input/
├── README.md
├── receptor/8V1Q_WT_UL30_DNA_no_water.pdb
├── ligands/docked_poses.sdf
├── affinity_predictions.csv
└── summary.json
```

`docked_poses.sdf`是保留三维坐标和键级的多记录 SDF。`affinity_predictions.csv`中的
`ligand_record_index`为 1-based 记录号，并包含 PNU 2D/Pharm2D、预筛3D、Smina、Docking
后3D、QED及理化字段；`predicted_paffinity`初始化为空。仓库内置的本地 3DMPG 模型可通过
`scripts/run_local_affinity.sh`完成逐 pose 预测、追踪校验和原子回填。预测中断时会保留
`model_pose_predictions.csv.partial`，用相同命令重跑即可断点续算；正式表更新前会备份为
`affinity_predictions.before_local_affinity.csv`。不要修改追踪键或把 PDBQT 当作 affinity
模型输入。

填写完成后运行独立的最终评分与 Top 200 导出命令：

```bash
bash scripts/run_final_score.sh
```

完整排名写入`runs/<run>/11_integrated_scoring/wt/ranked_parents.csv`，Top 200 写入
`runs/<run>/11_integrated_scoring/wt/top_200_hits.tsv`，列为
`hit_id、SMILES、Rank、score`。默认要求 affinity 的 pose 覆盖率和母体覆盖率均至少为95%，
空预测会直接报错，不会被静默填值。

## 质量控制

测试覆盖标准化与盐处理、参考结构校验、PNU 自相似性、配额与去重、单状态策略、Smina pose
追踪、PDB/SDF affinity 配对、空预测模板、单 WT 协议和最终评分字段。
