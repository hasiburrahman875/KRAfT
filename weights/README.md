# Weights

The W2C fold reproduction config expects the motion-model checkpoint at:

`experiments/uavswarm_ddm_1000_deeper/_epoch2100.pt`

The checkpoint file is about 149MB and exceeds normal GitHub file-size limits for direct source-file commits.

Use:

```bash
bash scripts/stage_w2c_weights.sh
```

Or provide an explicit source checkpoint:

```bash
bash scripts/stage_w2c_weights.sh /path/to/_epoch2100.pt
```
