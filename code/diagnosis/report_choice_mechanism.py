"""机制命题报告：从 dump_candidate_rollouts.py 的产物里做聚合统计 + 例图筛选。

命题 1（cifar10）：选 fd 1~2 的节点换来更多新前沿边特征。
  聚合：有 fd≥3 候选的步里，ΔE(被选) vs ΔE(当步最大 fd 候选) 的均值差与赢率；
        全局 P(chosen fd ≤ 2)。
  例图：max_fd ≥ 3、chosen fd ≤ 2、ΔE 差 ≥ 2、ev_correct，按 ΔE 差排序。

命题 2（collab）：选 fd 大的节点把揭示区锁进社区核。
  聚合：P(chosen fd = 当步最大 fd)；读对 vs 读错 episode 的 chosen fd；
        fd-max 步 vs 其他步的分类器熵降、翻对率、揭示子图密度跳升。
  例图：chosen fd = 当步最大、该步分类器翻对（pre 错 → post 对）、
        ev_correct，按熵降排序，n ≤ 500 保证拓扑可读。

用法（仓库根运行）:
  python code/diagnosis/report_choice_mechanism.py \
      --dump code/result/cifar10/experiments/ev-R-read/evidence/rl/behavior-analysis/candidate_rollouts_b10_g100s4.json
  python code/diagnosis/report_choice_mechanism.py \
      --dump code/result/collab/experiments/evidence1/evidence/rl/behavior-analysis/candidate_rollouts_b10_g100s1.json \
      --top 15
"""

import argparse
import json


def bootstrap_ci(vals, n=2000, seed=0):
    import random
    if not vals:
        return 0.0, 0.0
    rng = random.Random(seed)
    m = len(vals)
    means = sorted(sum(rng.choices(vals, k=m)) / m for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n)]


def cand_max_fd(rec):
    return max(c['fd'] for c in rec['candidates'])


def cand_max_fd_row(rec):
    return max(rec['candidates'], key=lambda c: c['fd'])


def chosen_row(rec):
    return next(c for c in rec['candidates'] if c['node'] == rec['chosen'])


def report_cifar10(d, top):
    eps = d['per_graph']
    steps = [(ep, rec) for ep in eps for rec in ep['steps']
             if not rec.get('teleport')]
    print(f"episodes={len(eps)}  policy_steps={len(steps)}  "
          f"ev_acc={d['ev_acc']}")

    fd_le2 = sum(1 for _, r in steps if chosen_row(r)['fd'] <= 2)
    print(f"\n[命题1 聚合] P(chosen fd ≤ 2) = {fd_le2 / len(steps):.4f} "
          f"({fd_le2}/{len(steps)})")
    dens = [r['revealed_edges_post'] / (r['revealed_count_pre'] + 1)
            for _, r in steps]
    print(f"揭示子图内边密度 = {sum(dens)/len(dens):.3f} edges/node"
          f"（对比 COLLAB ≈2.9：cifar10 = 扩张、COLLAB = 致密化）")
    ct_ch, ct_mean = [], []
    for _, r in steps:
        ch = chosen_row(r)
        ct_ch.append(ch['closed_tri'])
        ct_mean.append(ch['closed_tri']
                       - sum(c['closed_tri'] for c in r['candidates'])
                       / len(r['candidates']))
    print(f"闭合三角形 chosen = {sum(ct_ch)/len(ct_ch):.3f} "
          f"(vs 前沿平均 {sum(ct_mean)/len(ct_mean):+.3f})")

    mixed = [(ep, r) for ep, r in steps if cand_max_fd(r) >= 3]
    print(f"有 fd≥3 候选的步: {len(mixed)}"
          f"（这些步 reader 仍 0 次选 fd≥3）")
    d_chosen, d_maxfd, wins, ties, worse = [], [], 0, 0, 0
    d_vsmean = []
    for ep, r in mixed:
        ch = next(c for c in r['candidates'] if c['node'] == r['chosen'])
        mx = cand_max_fd_row(r)
        d_chosen.append(ch['new_fe'])
        d_maxfd.append(mx['new_fe'])
        d_vsmean.append(ch['new_fe'] - sum(c['new_fe'] for c in r['candidates'])
                        / len(r['candidates']))
        if ch['new_fe'] > mx['new_fe']:
            wins += 1
        elif ch['new_fe'] == mx['new_fe']:
            ties += 1
        else:
            worse += 1
    if mixed:
        lo_c, hi_c = bootstrap_ci(d_chosen)
        lo_m, hi_m = bootstrap_ci(d_maxfd)
        print(f"ΔE(chosen)   = {sum(d_chosen)/len(mixed):.3f} "
              f"[{lo_c:.3f}, {hi_c:.3f}]")
        print(f"ΔE(max-fd反事实) = {sum(d_maxfd)/len(mixed):.3f} "
              f"[{lo_m:.3f}, {hi_m:.3f}]")
        print(f"vs 前沿平均: +{sum(d_vsmean)/len(mixed):.3f}/步；"
              f"严格更优 {wins/len(mixed):.3f} / 并列 {ties/len(mixed):.3f} / "
              f"更差 {worse/len(mixed):.3f}")

    scored = []
    for ep, r in mixed:
        ch = next(c for c in r['candidates'] if c['node'] == r['chosen'])
        mx = cand_max_fd_row(r)
        diff = ch['new_fe'] - mx['new_fe']
        if ch['fd'] <= 2 and diff >= 2 and ep['ev_correct'] == 1:
            scored.append((diff, ch['prob'], ep, r, ch, mx))
    scored.sort(key=lambda t: (-t[0], -t[1], t[4]['node']))
    print(f"\n[命题1 例图候选] (diff≥2, chosen fd≤2, ev_correct；"
          f"共 {len(scored)} 步，top {top})")
    print("graph/seed/step  n   |F|  chosen(deg,fd,ΔE,p)  max-fd候选(deg,fd,ΔE)")
    for diff, _, ep, r, ch, mx in scored[:top]:
        print(f"g{ep['graph_idx']}/s{ep['seed_i']}/t{r['step']}  "
              f"n={ep['n_nodes']:<4} |F|={r['n_frontier']:<3} "
              f"chosen n.{ch['node']} ({ch['deg']},{ch['fd']},+{ch['new_fe']},"
              f"p={ch['prob']:.3f})  vs n.{mx['node']} ({mx['deg']},{mx['fd']},"
              f"+{mx['new_fe']})  diff=+{diff}")


def report_collab(d, top):
    eps = d['per_graph']
    steps = [(ep, rec) for ep in eps for rec in ep['steps']
             if not rec.get('teleport')]
    print(f"episodes={len(eps)}  policy_steps={len(steps)}  "
          f"ev_acc={d['ev_acc']}")

    eq = sum(1 for _, r in steps
             if chosen_row(r)['fd'] == cand_max_fd(r))
    print(f"\n[命题2 聚合] P(chosen fd = 当步最大 fd) = {eq/len(steps):.4f} "
          f"({eq}/{len(steps)})")
    dens = [r['revealed_edges_post'] / (r['revealed_count_pre'] + 1)
            for _, r in steps]
    print(f"揭示子图内边密度 = {sum(dens)/len(dens):.3f} edges/node"
          f"（≈3×树状行走，揭示区是致密团 = 社区核）")
    ct_ch, ct_mean = [], []
    for _, r in steps:
        ch = chosen_row(r)
        ct_ch.append(ch['closed_tri'])
        ct_mean.append(ch['closed_tri']
                       - sum(c['closed_tri'] for c in r['candidates'])
                       / len(r['candidates']))
    print(f"闭合三角形 chosen = {sum(ct_ch)/len(ct_ch):.2f} "
          f"(vs 前沿平均 {sum(ct_mean)/len(ct_mean):+.2f}；"
          f"cifar10 对照值见 cifar10 报告)")
    if 'clf_entropy_pre' in steps[0][1]:
        drops = [r['clf_entropy_post'] - r['clf_entropy_pre']
                 for _, r in steps
                 if r['clf_entropy_pre'] is not None
                 and r['clf_entropy_post'] is not None]
        flips = sum(1 for ep, r in steps
                    if r['clf_pred_pre'] != ep['label']
                    and r['clf_pred_post'] == ep['label'])
        print(f"分类器置信动态（无信号备查）: ΔH(clf) = "
              f"{sum(drops)/max(len(drops),1):+.4f}，"
              f"该步翻对率 = {flips/len(steps):.4f}")

    scored = []
    for ep, r in steps:
        ch = chosen_row(r)
        if ep['ev_correct'] != 1 or ep['n_nodes'] > 500 or ch['fd'] == 0:
            continue
        dens_pre = r['revealed_edges_pre'] / r['revealed_count_pre']
        dens_post = r['revealed_edges_post'] / (r['revealed_count_pre'] + 1)
        scored.append((-ch['closed_tri'], ep, r, ch, dens_pre, dens_post))
    scored.sort(key=lambda t: (t[0], t[1]['graph_idx'], t[2]['step']))
    print(f"\n[命题2 例图候选] (chosen fd=max、ev_correct、n≤500；"
          f"按闭合三角形降序，top {top})")
    print("graph/seed/step  n   B   chosen(deg,fd,ΔE,△闭合,p)  "
          "密度pre→post")
    for neg_ct, ep, r, ch, dp, dq in scored[:top]:
        print(f"g{ep['graph_idx']}/s{ep['seed_i']}/t{r['step']}  "
              f"n={ep['n_nodes']:<4} B={ep['budget']:<3} "
              f"n.{ch['node']} ({ch['deg']},{ch['fd']},+{ch['new_fe']},"
              f"{ch['closed_tri']},{ch['prob']:.3f})  {dp:.3f}→{dq:.3f}")


def main():
    parser = argparse.ArgumentParser(description='机制命题报告')
    parser.add_argument('--dump', required=True)
    parser.add_argument('--top', type=int, default=10)
    args = parser.parse_args()
    with open(args.dump, encoding='utf-8') as f:
        d = json.load(f)
    print(f"dump: {args.dump}")
    print(f"checkpoint: {d['config']['checkpoint']}")
    if d['config'].get('clf_entropy'):
        report_collab(d, args.top)
    else:
        report_cifar10(d, args.top)


if __name__ == '__main__':
    main()
