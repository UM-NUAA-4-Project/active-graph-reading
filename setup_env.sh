#!/usr/bin/env bash
# =============================================================================
# 一键环境搭建脚本 — Ubuntu 服务器（conda 环境 gnn）
#
# 执行内容:
#   1. Ubuntu apt 换源 (阿里云, 自动识别版本代号)
#   2. 安装 tmux + curl
#   3. pip 换源 (阿里云)
#   4. conda 换源 (运行时探测可达镜像: NJU/USTC/SJTU/TUNA, 并同步 environment-linux.yml)
#   5. 创建 conda 环境 gnn (code/environment-linux.yml)
#   6. 验证环境 (import torch / dgl / ogb / torch_geometric)
#
# 用法:
#   bash setup_env.sh
#
# 前提:
#   - Ubuntu，当前用户有 sudo 权限
#   - 已安装 Miniconda / Anaconda
# 注: 针对 22.04 (jammy) 验证过；其他版本自动替换代号。24.04+ 使用 deb822
#     格式 (sources.list.d/ubuntu.sources)，本脚本写入经典 sources.list 后
#     两者会并存，建议手动清空 ubuntu.sources 以免 apt 仍访问旧源。
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_YML="$SCRIPT_DIR/code/environment-linux.yml"

log()  { echo -e "\033[1;32m[OK]\033[0m   $*"; }
info() { echo -e "\033[1;34m[..]\033[0m   $*"; }
die()  { echo -e "\033[1;31m[FAIL]\033[0m $*" >&2; exit 1; }

# ---- 0. 前置检查 ----
[[ "$(uname -s)" == "Linux" ]] || die "仅支持 Linux 服务器"
sudo -v || die "sudo 校验失败（当前用户需有 sudo 权限）"

# ---- 1. Ubuntu apt 换源（阿里云）----
CODENAME=$(sed -n 's/^VERSION_CODENAME=//p' /etc/os-release | tr -d '"' | head -n1)
[[ -n "$CODENAME" ]] || die "无法从 /etc/os-release 识别 Ubuntu 版本代号"
info "Ubuntu 代号: ${CODENAME} → 写入阿里云源"

if [[ -f /etc/apt/sources.list ]]; then
    BACKUP="/etc/apt/sources.list.bak.$(date +%Y%m%d-%H%M%S)"
    sudo cp -a /etc/apt/sources.list "$BACKUP"
    log "原 sources.list 已备份至 $BACKUP"
fi

sudo tee /etc/apt/sources.list > /dev/null <<EOF
deb https://mirrors.aliyun.com/ubuntu/ ${CODENAME} main restricted universe multiverse
deb https://mirrors.aliyun.com/ubuntu/ ${CODENAME}-security main restricted universe multiverse
deb https://mirrors.aliyun.com/ubuntu/ ${CODENAME}-updates main restricted universe multiverse
deb https://mirrors.aliyun.com/ubuntu/ ${CODENAME}-backports main restricted universe multiverse
EOF
log "sources.list 已写入"

info "apt-get update ..."
sudo apt-get update
log "apt 源更新完成"

# ---- 2. 安装 tmux + curl ----
info "安装 tmux curl ..."
sudo apt-get install -y tmux curl
log "tmux/curl 安装完成"

# ---- 3. pip 换源（阿里云；清华源在部分机房不可达，与 apt/conda 源保持同家）----
mkdir -p "$HOME/.config/pip"
cat > "$HOME/.config/pip/pip.conf" <<'EOF'
[global]
index-url = https://mirrors.aliyun.com/pypi/simple/
trusted-host = mirrors.aliyun.com
EOF
log "pip 源已设为阿里云 ($HOME/.config/pip/pip.conf)"

# ---- 4. 定位 conda ----
CONDA_ROOT=""
if command -v conda >/dev/null 2>&1 && [[ -x "$(command -v conda)" ]]; then
    CONDA_ROOT="$(cd "$(dirname "$(readlink -f "$(command -v conda)")")/.." && pwd)"
fi
if [[ -z "$CONDA_ROOT" ]]; then
    for cand in "$HOME/miniconda3" "$HOME/anaconda3" /opt/miniconda3 /opt/anaconda3 /opt/conda \
                /home/*/miniconda3 /home/*/anaconda3; do
        if [[ -x "$cand/bin/conda" ]]; then
            CONDA_ROOT="$cand"
            break
        fi
    done
fi
[[ -n "$CONDA_ROOT" ]] || die "未找到 conda。请先安装 Miniconda（阿里云镜像: https://mirrors.aliyun.com/anaconda/miniconda/）"
source "$CONDA_ROOT/etc/profile.d/conda.sh" 2>/dev/null || die "无法加载 $CONDA_ROOT/etc/profile.d/conda.sh"
command -v conda >/dev/null 2>&1 || die "conda 加载失败"
log "conda 根目录: $CONDA_ROOT"

# ---- 5. conda 换源（运行时探测可达镜像）----
# 背景：conda env create -f 会把 .condarc 与 yml 的 channels 取并集，任何一路
# 不可达（HTTP 000，机房差异：如阿里云 anaconda 树已下线、BFSU 重定向回清华、
# 部分主机连不上清华）整个 solve 即失败；具名源（pytorch/conda-forge/defaults）
# 还会走 repo.anaconda.com 外网。因此：探测选一个本机可达的镜像 base，.condarc
# 与 yml（临时副本）统一换成同一套 URL。
# 候选序 = 实测直连性（NJU 全直连 > USTC conda-forge 直连/另两路 302→NJU >
# SJTU 302→jcloud OSS > TUNA；BFSU 302→TUNA 与清华同命，不列）。
pick_conda_mirror() {
    local base cf main
    for base in \
        https://mirror.nju.edu.cn/anaconda \
        https://mirrors.ustc.edu.cn/anaconda \
        https://mirror.sjtu.edu.cn/anaconda \
        https://mirrors.tuna.tsinghua.edu.cn/anaconda; do
        cf=$(curl -sL -o /dev/null -w '%{http_code}' -r 0-99 -m 10 \
             "$base/cloud/conda-forge/noarch/repodata.json" 2>/dev/null || echo 000)
        main=$(curl -sL -o /dev/null -w '%{http_code}' -r 0-99 -m 10 \
               "$base/pkgs/main/noarch/repodata.json" 2>/dev/null || echo 000)
        if [[ "$cf" == 20* && "$main" == 20* ]]; then
            echo "$base"; return 0
        fi
        info "镜像不可达，换下一个: $base (conda-forge=$cf pkgs/main=$main)"
    done
    return 1
}

info "探测可用的 conda 镜像 ..."
MIRROR_BASE="$(pick_conda_mirror)" \
    || die "所有候选 conda 镜像均不可达。请手动配置 ~/.condarc（三路 channels：任一可达镜像的 /cloud/pytorch、/cloud/conda-forge、/pkgs/main）后重跑"
log "conda 镜像选定: $MIRROR_BASE"

info "配置 conda 源 ..."
conda config --remove-key channels 2>/dev/null || true  # 先清空，重复执行不产生重复条目
conda config --add channels "$MIRROR_BASE/pkgs/main/"
conda config --add channels "$MIRROR_BASE/cloud/conda-forge/"
conda config --add channels "$MIRROR_BASE/cloud/pytorch/"
conda config --set show_channel_urls yes
log "conda 源已配置（优先级: pytorch > conda-forge > main，base=$MIRROR_BASE）"

# ---- 6. 创建 conda 环境 ----
[[ -f "$ENV_YML" ]] || die "环境文件不存在: $ENV_YML"
if conda env list | awk '{print $1}' | grep -qx 'gnn'; then
    log "环境 gnn 已存在，跳过创建（如需重建: conda env remove -n gnn 后重跑本脚本）"
else
    # yml 缺省 base（NJU）与探测结果不一致时生成临时副本替换，确保 env create
    # 用的三路 channel 与 .condarc 完全一致（避免并集拉到不可达源）
    ENV_YML_USE="$ENV_YML"
    YML_DEFAULT_BASE="https://mirror.nju.edu.cn/anaconda"
    if [[ "$MIRROR_BASE" != "$YML_DEFAULT_BASE" ]]; then
        ENV_YML_USE="$(mktemp --suffix=.yml)"
        sed "s|$YML_DEFAULT_BASE|$MIRROR_BASE|g" "$ENV_YML" > "$ENV_YML_USE"
        log "yml channels 已替换为 $MIRROR_BASE（临时副本: $ENV_YML_USE）"
    fi
    # 外源可达性预检：pip 只剩 download.pytorch.org（torch+cu121 轮子）；
    # dgl 走 conda dglteam/label/th23_cu121（data.dgl.ai 对 CN 网络不可达）
    curl -fsSL -o /dev/null -m 10 "https://download.pytorch.org/whl/cu121/" 2>/dev/null \
        || echo -e "\033[1;33m[WARN]\033[0m download.pytorch.org 不可达（pip 装 torch+cu121 会失败）"
    curl -fsSL -o /dev/null -m 15 \
        "https://conda.anaconda.org/dglteam/label/th23_cu121/linux-64/repodata.json" 2>/dev/null \
        || cat <<'WEOF'
[WARN] conda.anaconda.org/dglteam 不可达：dgl==2.2.1 将安装失败。
       离线兜底：任一可达机器执行
         curl -LO https://conda.anaconda.org/dglteam/label/th23_cu121/linux-64/dgl-2.2.1.th23.cu121-py310_0.tar.bz2
       scp 到服务器后：conda install -n gnn ./dgl-2.2.1.th23.cu121-py310_0.tar.bz2
WEOF
    info "创建 conda 环境 gnn（首次下载数 GB，耗时较长）..."
    conda env create -f "$ENV_YML_USE"
    log "环境 gnn 创建完成"
    [[ "$ENV_YML_USE" == "$ENV_YML" ]] || rm -f "$ENV_YML_USE"
fi

# ---- 7. 验证环境 ----
info "验证环境（import 检查）..."
if conda run -n gnn python - <<'EOF'
import torch, dgl, ogb, torch_geometric
print(f"torch={torch.__version__}  cuda_available={torch.cuda.is_available()}")
print(f"dgl={dgl.__version__}  ogb={ogb.__version__}  pyg={torch_geometric.__version__}")
EOF
then
    log "环境验证通过"
else
    die "环境验证失败。常见原因：上次创建时 pip 段失败残留了半成品环境（conda 包在、"
        "torch/dgl 缺失），本次被「已存在，跳过创建」跳过。修复："
        "conda env remove -n gnn -y 后重跑本脚本（pip 缓存仍在，torch 不会重下）"
fi

echo
log "全部完成。激活环境: conda activate gnn"
