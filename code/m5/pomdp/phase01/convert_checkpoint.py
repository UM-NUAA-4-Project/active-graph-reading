"""checkpoint 结构转换脚本：旧 GSAE (带 encoder/decoder) → 新 GSAE (不含 encoder/decoder)。

旧结构:
  GSAE:        encoder.weight(128,300) + encoder.bias(128)
               decoder.weight(300,128) + decoder.bias(300)
  EvidenceNet: evidence_head.0.weight(37,128) + evidence_head.0.bias(37)

新结构:
  GSAE:        无 encoder/decoder（GNN + attention pool 直接出 v ∈ R^300）
  EvidenceNet: evidence_head.0 (Linear 300→128, 吸收旧 encoder)
               evidence_head.2 (Linear 128→37, 原 evidence_head.0)

用法：
  python convert_checkpoint.py --input checkpoint_epoch1.pt --output checkpoint_epoch1_v2.pt
  python convert_checkpoint.py --input checkpoint_epoch2.pt  # 自动命名 *_v2.pt
"""

import argparse
import sys
from pathlib import Path


def convert_checkpoint(input_path: Path, output_path: Path):
    import torch

    ckpt = torch.load(str(input_path), map_location='cpu', weights_only=False)
    old_gsae = ckpt['gsae_state_dict']
    old_ev = ckpt['evidence_state_dict']

    # 1. GSAE: 移除 encoder.* 和 decoder.*
    new_gsae = {k: v for k, v in old_gsae.items()
                if not k.startswith('encoder.') and not k.startswith('decoder.')}

    removed = [k for k in old_gsae if k not in new_gsae]
    print(f"  Removed from GSAE: {removed}")

    # 2. EvidenceNet: 吸收 encoder，形成两层投影
    new_ev = {}
    new_ev['evidence_head.0.weight'] = old_gsae['encoder.weight']          # 128x300
    new_ev['evidence_head.0.bias'] = old_gsae['encoder.bias']              # 128
    new_ev['evidence_head.2.weight'] = old_ev['evidence_head.0.weight']    # 37x128
    new_ev['evidence_head.2.bias'] = old_ev['evidence_head.0.bias']        # 37

    print(f"  evidence_head.0.weight: {list(new_ev['evidence_head.0.weight'].shape)}")
    print(f"  evidence_head.0.bias:   {list(new_ev['evidence_head.0.bias'].shape)}")
    print(f"  evidence_head.2.weight: {list(new_ev['evidence_head.2.weight'].shape)}")
    print(f"  evidence_head.2.bias:   {list(new_ev['evidence_head.2.bias'].shape)}")

    # 3. 更新 config
    cfg = ckpt['config'].copy()
    cfg.pop('lambda_edl', None)
    cfg.pop('lambda_kl', None)

    # 4. 保存
    new_ckpt = {
        'epoch': ckpt['epoch'],
        'gsae_state_dict': new_gsae,
        'evidence_state_dict': new_ev,
        'optimizer_state_dict': ckpt['optimizer_state_dict'],
        'training_log': ckpt.get('training_log', []),
        'config': cfg,
    }
    torch.save(new_ckpt, str(output_path))
    print(f"  Saved: {output_path}")


def verify_checkpoint(checkpoint_path: Path):
    """加载新 checkpoint 到模型验证 b+u=1。"""
    import sys
    _repo_root = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(_repo_root))

    import torch
    from active_graph_reader.reader.gsae import GSAE
    from active_graph_reader.reader.evidence_network import EvidenceNetwork

    ckpt = torch.load(str(checkpoint_path), map_location='cpu', weights_only=False)
    cfg = ckpt['config']

    gsae = GSAE(edge_feat_dim=7, hidden_dim=cfg['hidden_dim'], k_state=128,
                n_layers=2, dropout=0.5)
    gsae.load_state_dict(ckpt['gsae_state_dict'])

    ev = EvidenceNetwork(hidden_dim=cfg['hidden_dim'], k_evidence=37)
    ev.load_state_dict(ckpt['evidence_state_dict'])

    v = torch.randn(1, cfg['hidden_dim'])
    out = ev(v)
    assert out.e.shape == (1, 37), f"unexpected e shape {out.e.shape}"
    assert (out.e >= 0).all(), "evidence must be non-negative"
    assert torch.allclose(out.b.sum(dim=-1) + out.u, torch.ones(1), atol=1e-5), \
        f"b+u != 1, got {out.b.sum(dim=-1) + out.u}"

    g_params = sum(p.numel() for p in gsae.parameters())
    e_params = sum(p.numel() for p in ev.parameters())
    print(f"  Verification OK: GSAE={g_params:,} + EvidenceNet={e_params:,} = {g_params+e_params:,} params")


def main():
    parser = argparse.ArgumentParser(description='Convert old GSAE+Evidence checkpoint to new structure')
    parser.add_argument('--input', required=True, help='旧 checkpoint 路径')
    parser.add_argument('--output', default='', help='输出路径（默认：input 加 _v2 后缀）')
    parser.add_argument('--no-verify', action='store_true', help='跳过验证')
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: {input_path} not found")
        sys.exit(1)

    if args.output:
        output_path = Path(args.output)
    else:
        stem = input_path.stem
        output_path = input_path.with_stem(stem + '_v2')

    print(f"Converting: {input_path} -> {output_path}")
    convert_checkpoint(input_path, output_path)

    if not args.no_verify:
        print("Verifying...")
        verify_checkpoint(output_path)

    print("Done.")


if __name__ == '__main__':
    main()
