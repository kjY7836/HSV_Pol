"""
分子处理工具函数
"""
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import torch


# ==================== 节点类型字典 ====================
# 常见原子类型（one-hot 编码索引）
# 按照在有机分子中的常见程度和重要性排序
ATOM_TYPE_DICT = {
    'H': 0,    # 氢
    'C': 1,    # 碳
    'N': 2,    # 氮
    'O': 3,    # 氧
    'F': 4,    # 氟
    'P': 5,    # 磷
    'S': 6,    # 硫
    'Cl': 7,   # 氯
    'Br': 8,   # 溴
    'I': 9,    # 碘
    'B': 10,   # 硼
    'Si': 11,  # 硅
    'Se': 12,  # 硒
    'As': 13,  # 砷
    'Li': 14,  # 锂
    'Na': 15,  # 钠
    'K': 16,   # 钾
    'Mg': 17,  # 镁
    'Ca': 18,  # 钙
    'Zn': 19,  # 锌
    'Fe': 20,  # 铁
    'Cu': 21,  # 铜
    'Mn': 22,  # 锰
    'Co': 23,  # 钴
    'Ni': 24,  # 镍
    'OTHER': 25,  # 其他原子类型
    'virtual_intra_node': 26,   # 虚拟内部节点（虚拟全局节点）
    'virtual_inter_node': 27,   # 虚拟跨片段节点
    'MASK': 28,   # Mask token
}

# 原子序数到类型的映射
ATOMIC_NUM_TO_TYPE = {
    1: 0,    # H
    6: 1,    # C
    7: 2,    # N
    8: 3,    # O
    9: 4,    # F
    15: 5,   # P
    16: 6,   # S
    17: 7,   # Cl
    35: 8,   # Br
    53: 9,   # I
    5: 10,   # B
    14: 11,  # Si
    34: 12,  # Se
    33: 13,  # As
    3: 14,   # Li
    11: 15,  # Na
    19: 16,  # K
    12: 17,  # Mg
    20: 18,  # Ca
    30: 19,  # Zn
    26: 20,  # Fe
    29: 21,  # Cu
    25: 22,  # Mn
    27: 23,  # Co
    28: 24,  # Ni
}

# 节点类型总数（常见原子类型 + 其他 + 虚拟节点类型）
NUM_ATOM_TYPES = len(ATOM_TYPE_DICT)  # 28

# 节点特征总维度 = 原子类型(28)，虚拟节点已融合到原子类型中
NODE_FEAT_DIM = NUM_ATOM_TYPES  # 28


# ==================== 边类型字典 ====================
# 化学键类型（one-hot 编码索引）
BOND_TYPE_DICT = {
    'SINGLE': 0,      # 单键
    'DOUBLE': 1,      # 双键
    'TRIPLE': 2,      # 三键
    'AROMATIC': 3,    # 芳香键
    'virtual_intra_edge': 4,   # 虚拟内部边（intraedge）
    'virtual_inter_edge': 5,    # 虚拟跨片段边（interface）
}

# RDKit 键类型到字典索引的映射
RDKIT_BOND_TYPE_TO_INDEX = {
    1: 0,   # SINGLE
    2: 1,   # DOUBLE
    3: 2,   # TRIPLE
    12: 3,  # AROMATIC
}

# 边类型总数
NUM_BOND_TYPES = len(BOND_TYPE_DICT)  # 6

# 边特征总维度 = 键类型(6) + 距离(1) = 7
EDGE_FEAT_DIM = NUM_BOND_TYPES + 1


def mol_to_coords(mol):
    """
    从 RDKit 分子对象获取 3D 坐标
    
    Args:
        mol: RDKit 分子对象
    
    Returns:
        coords: (N, 3) numpy 数组，原子坐标
    """
    conf = mol.GetConformer()
    coords = np.array([conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms())])
    return coords.astype(np.float32)


def get_atom_features(atom, is_virtual_intra_node=False, is_virtual_inter_node=False):
    """
    获取原子特征（类别索引）
    
    Args:
        atom: RDKit 原子对象（如果为 None，则返回虚拟节点特征）
        is_virtual_intra_node: 是否是虚拟内部节点（虚拟全局节点）
        is_virtual_inter_node: 是否是虚拟跨片段节点
    
    Returns:
        atom_type_idx: 原子类型索引（标量），范围 [0, NUM_ATOM_TYPES-1]
    """
    if atom is None:
        # 虚拟节点：根据标志设置对应的虚拟节点类型
        if is_virtual_intra_node:
            return ATOM_TYPE_DICT['virtual_intra_node']
        elif is_virtual_inter_node:
            return ATOM_TYPE_DICT['virtual_inter_node']
        else:
            # 默认使用虚拟内部节点
            return ATOM_TYPE_DICT['virtual_intra_node']
    else:
        atomic_num = atom.GetAtomicNum()
        if atomic_num in ATOMIC_NUM_TO_TYPE:
            return ATOMIC_NUM_TO_TYPE[atomic_num]
        else:
            # 其他原子类型
            return ATOM_TYPE_DICT['OTHER']


def get_bond_features(bond, is_virtual_intra_edge=False, is_virtual_inter_edge=False):
    """
    获取化学键特征（类别索引）
    
    Args:
        bond: RDKit 键对象（如果为 None，则返回虚拟边特征）
        is_virtual_intra_edge: 是否是虚拟内部边（intraedge）
        is_virtual_inter_edge: 是否是虚拟跨片段边（interface）
    
    Returns:
        bond_type_idx: 键类型索引（标量），范围 [0, NUM_BOND_TYPES-1]
    """
    if bond is None:
        # 虚拟边
        if is_virtual_intra_edge:
            return BOND_TYPE_DICT['virtual_intra_edge']
        elif is_virtual_inter_edge:
            return BOND_TYPE_DICT['virtual_inter_edge']
        else:
            # 默认使用虚拟内部边
            return BOND_TYPE_DICT['virtual_intra_edge']
    else:
        bond_type = int(bond.GetBondType())
        if bond_type in RDKIT_BOND_TYPE_TO_INDEX:
            return RDKIT_BOND_TYPE_TO_INDEX[bond_type]
        else:
            # 其他类型，默认返回单键
            return BOND_TYPE_DICT['SINGLE']


def ensure_3d_coords(obj):
    """
    兼容 RDKit Mol 与 numpy/torch 坐标的 3D 坐标校验：
    - RDKit Mol：若无构象则生成一个 3D 构象，返回 Mol
    - numpy/torch 坐标：确保形状为 (N, 3)，必要时从一维 [3N] reshape，返回 numpy 数组
    """
    # RDKit Mol 分支
    if hasattr(obj, "GetNumConformers"):
        mol = obj
        if mol.GetNumConformers() == 0:
            AllChem.EmbedMolecule(mol, randomSeed=42)
            AllChem.MMFFOptimizeMolecule(mol)
        return mol
    
    # 坐标分支
    coords = obj
    if isinstance(coords, torch.Tensor):
        coords = coords.detach().cpu().numpy()
    try:
        coords = np.asarray(coords, dtype=np.float32)
    except Exception:
        return None
    
    if coords.ndim == 1:
        if coords.size % 3 != 0:
            return None
        coords = coords.reshape(-1, 3)
    elif coords.ndim == 2 and coords.shape[1] == 3:
        pass
    else:
        return None
    
    return coords


def center_coordinates(coords):
    """
    坐标去中心化（提取为独立函数，避免重复代码）
    
    Args:
        coords: numpy array of shape (N, 3) or list/tuple or torch.Tensor
    
    Returns:
        coords_centered: numpy array of shape (N, 3), 去中心化后的坐标
    """
    if isinstance(coords, torch.Tensor):
        coords = coords.numpy()
    coords = np.asarray(coords, dtype=np.float32)
    if coords.ndim == 2 and coords.shape[1] == 3:
        return coords - coords.mean(axis=0)
    return coords
