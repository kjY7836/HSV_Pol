# Source provenance

The model implementation bundled here was copied from the clean local checkout
of `https://github.com/kjY7836/3DMPG.git` at commit
`05cda2aec43deb0513883aedc829de88eba1b2fc`.

Copied files:

- `affinityV2/model.py`
- `affinityV2/data.py`
- `models/molgnet3d.py`
- `utils/molecular_utils.py`
- their package `__init__.py` files

Checkpoint:

- file: `weights/affinity_v2_r2020_semantic_epoch3encoder_e90_1_rerun.pt`
- SHA-256: `4d950e8d751cc962c9d3b15bc40c5ef346b0bff95656db94065df7b6e7e6ad0d`
- checkpoint epoch: 85
- saved validation RMSE: 1.125950813293457 pK units
- label mean/std: 6.382334430320695 / 1.8218264307708651

Bundled receptor assets:

- `receptors/8V1Q_WT_UL30_DNA_no_water.pdb` SHA-256:
  `eba3082bda16f5ff86e8597910c473c4dede2f1da7fba432583a871a64fb9c91`
- `receptors/8V1Q_WT_UL30_DNA_no_water.pdbqt` SHA-256:
  `12496b8459d857a0aa17d6423493fcf4f4527c8f5be9363902a240d2c0fc352a`

The inference wrapper uses checkpoint-saved architecture and normalization
metadata. It does not import from the reference `3DMPG` checkout.
