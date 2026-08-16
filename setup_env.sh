#!/usr/bin/env bash
# =============================================================================
# 一键环境搭建脚本 — Ubuntu 服务器（conda 环境 gnn）
#
# 执行内容:
#   1. Ubuntu apt 换源 (阿里云, 自动识别版本代号)
#   2. 安装 tmux
#   3. pip 换源 (清华)
#   4. conda 换源 (清华)
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

# ---- 2. 安装 tmux ----
info "安装 tmux ..."
sudo apt-get install -y tmux
log "tmux 安装完成"

# ---- 3. pip 换源（清华）----
mkdir -p "$HOME/.config/pip"
cat > "$HOME/.config/pip/pip.conf" <<'EOF'
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
trusted-host = pypi.tuna.tsinghua.edu.cn
EOF
log "pip 源已设为清华 ($HOME/.config/pip/pip.conf)"

# ---- 4. 定位 conda ----
CONDA_ROOT=""
if command -v conda >/dev/null 2>&1 && [[ -x "$(command -v conda)" ]]; then
    CONDA_ROOT="$(cd "$(dirname "$(readlink -f "$(command -v conda)")")/.." && pwd)"
fi
if [[ -z "$CONDA_ROOT" ]]; then
    for cand in "$HOME/miniconda3" "$HOME/anaconda3" /opt/miniconda3 /opt/anaconda3; do
        if [[ -x "$cand/bin/conda" ]]; then
            CONDA_ROOT="$cand"
            break
        fi
    done
fi
[[ -n "$CONDA_ROOT" ]] || die "未找到 conda。请先安装 Miniconda（清华镜像: https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/）"
source "$CONDA_ROOT/etc/profile.d/conda.sh" 2>/dev/null || die "无法加载 $CONDA_ROOT/etc/profile.d/conda.sh"
command -v conda >/dev/null 2>&1 || die "conda 加载失败"
log "conda 根目录: $CONDA_ROOT"

# ---- 5. conda 换源（清华）----
info "配置 conda 清华源 ..."
conda config --remove-key channels 2>/dev/null || true  # 先清空，重复执行不产生重复条目
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
conda config --set show_channel_urls yes
log "conda 源已配置（优先级: pytorch > conda-forge > free > main）"

# ---- 6. 创建 conda 环境 ----
[[ -f "$ENV_YML" ]] || die "环境文件不存在: $ENV_YML"
if conda env list | awk '{print $1}' | grep -qx 'gnn'; then
    log "环境 gnn 已存在，跳过创建（如需重建: conda env remove -n gnn 后重跑本脚本）"
else
    info "创建 conda 环境 gnn（首次下载数 GB，耗时较长）..."
    conda env create -f "$ENV_YML"
    log "环境 gnn 创建完成"
fi

# ---- 7. 验证环境 ----
info "验证环境（import 检查）..."
conda run -n gnn python - <<'EOF'
import torch, dgl, ogb, torch_geometric
print(f"torch={torch.__version__}  cuda_available={torch.cuda.is_available()}")
print(f"dgl={dgl.__version__}  ogb={ogb.__version__}  pyg={torch_geometric.__version__}")
EOF
log "环境验证通过"

echo
log "全部完成。激活环境: conda activate gnn"
