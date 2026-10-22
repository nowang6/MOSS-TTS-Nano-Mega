#!/usr/bin/env bash
#
# 在 macOS (Apple Silicon) 上安装 MOSS-TTS-Nano 推理所需的原生依赖。
#
#   bash scripts/install_macos_deps.sh
#
# 装什么：
#   1. openfst        —— brew 装的 pynini 编译依赖（PyPI 的 pynini 没有任何 macOS wheel）
#   2. pynini         —— 源码编译（2.1.7；2.1.6 与 OpenFST 1.8 不兼容）
#   3. WeTextProcessing —— 文本正则化，--no-deps 安装以保住 pynini 2.1.7
#   4. soundfile      —— torchaudio 在 macOS 没有 ffmpeg 后端，靠它才能读 wav
#
# 环境变量：
#   PYNI_ROOT           PyPI 源站根目录（默认清华镜像）
#   PYNINI_VERSION      pynini 版本（默认 2.1.7）
#   WETEXT_VERSION      WeTextProcessing 版本（默认 1.0.4.1）
#   OPENFST_PREFIX      openfst 前缀（默认由 brew 决定）
#   VENV_PY             目标虚拟环境的 python（默认 .venv/bin/python）
#   SKIP_OPENFST=1      跳过 brew install openfst
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PY="${VENV_PY:-$REPO_ROOT/.venv/bin/python}"
WORK_DIR="$(mktemp -d)"

PYPI_ROOT="${PYPI_ROOT:-https://pypi.tuna.tsinghua.edu.cn}"
PYNINI_VERSION="${PYNINI_VERSION:-2.1.7}"
WETEXT_VERSION="${WETEXT_VERSION:-1.0.4.1}"

log() { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[!]\033[0m %s\n' "$*" >&2; }
die() { printf '\033[1;31m[x]\033[0m %s\n' "$*" >&2; exit 1; }

cleanup() { rm -rf "$WORK_DIR"; }
trap cleanup EXIT

# ---------------------------------------------------------------- 0. 环境检查
[[ "$(uname -s)" == "Darwin" ]] || die "本脚本只适用于 macOS"
[[ -x "$VENV_PY" ]] || die "找不到 $VENV_PY，先执行: uv venv .venv && uv pip install -r requirements.txt"

# 安装器：优先用 uv（仓库约定），没有就退到 venv 自带的 pip
if command -v uv >/dev/null 2>&1; then
  installer=(uv pip install --python "$VENV_PY")
elif "$VENV_PY" -m pip --version >/dev/null 2>&1; then
  installer=("$VENV_PY" -m pip install)
else
  "$VENV_PY" -m ensurepip --upgrade >/dev/null 2>&1 || die "venv 里没有 pip，且 ensurepip 失败"
  installer=("$VENV_PY" -m pip install)
fi
log "安装器: ${installer[*]}"

vpy_has() { "$VENV_PY" -c "import $1" >/dev/null 2>&1; }

# ---------------------------------------------------------------- 1. openfst
if [[ "${SKIP_OPENFST:-0}" != "1" ]]; then
  if command -v brew >/dev/null 2>&1; then
    if ! brew list --versions openfst >/dev/null 2>&1; then
      log "brew install openfst"
      brew install openfst
    else
      log "openfst 已安装: $(brew list --versions openfst)"
    fi
    OPENFST_PREFIX="${OPENFST_PREFIX:-$(brew --prefix openfst)}"
  else
    [[ -n "${OPENFST_PREFIX:-}" ]] || die "没有 brew，请手动安装 openfst 并设置 OPENFST_PREFIX"
    log "跳过 brew，使用 OPENFST_PREFIX=$OPENFST_PREFIX"
  fi
else
  [[ -n "${OPENFST_PREFIX:-}" ]] || OPENFST_PREFIX="$(brew --prefix openfst 2>/dev/null || true)"
fi
[[ -d "$OPENFST_PREFIX/include/fst" ]] || die "openfst 头文件缺失: $OPENFST_PREFIX/include/fst"

# ---------------------------------------------------------------- 2. pynini
if vpy_has pynini; then
  log "pynini 已安装: $("$VENV_PY" -c 'import pynini; print(pynini.__version__)')"
else
  log "下载 pynini ${PYNINI_VERSION} 源码包"
  simple_index="$PYPI_ROOT/simple/pynini/"
  href="$(curl -fsSL "$simple_index" \
    | grep -oE "href=\"[^\"]*pynini-${PYNINI_VERSION}\.tar\.gz[^\"]*\"" \
    | head -1 | sed -e 's/^href="//' -e 's/"$//' -e 's/#.*$//')"
  [[ -n "$href" ]] || die "在 $simple_index 找不到 pynini-${PYNINI_VERSION}.tar.gz"

  case "$href" in
    http*) sdist_url="$href" ;;
    # 镜像返回的是相对路径（../../packages/...）
    *) sdist_url="$PYPI_ROOT/${href#../../}" ;;
  esac
  log "sdist: $sdist_url"
  curl -fsSL -o "$WORK_DIR/pynini.tar.gz" "$sdist_url"
  tar xzf "$WORK_DIR/pynini.tar.gz" -C "$WORK_DIR"
  src_dir="$WORK_DIR/pynini-${PYNINI_VERSION}"

  # pynini 的 setup.py 把 macOS 部署目标硬编码成 10.12，
  # 而 std::optional 需要 10.14+，否则编译报 "'value' is unavailable"。
  if grep -q -- '-mmacosx-version-min=10\.12' "$src_dir/setup.py"; then
    log "patch setup.py: -mmacosx-version-min=10.12 -> 11.0"
    sed -i '' 's/-mmacosx-version-min=10\.12/-mmacosx-version-min=11.0/' "$src_dir/setup.py"
  fi

  log "编译 pynini ${PYNINI_VERSION}（约 2-3 分钟）"
  CPLUS_INCLUDE_PATH="$OPENFST_PREFIX/include${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}" \
  C_INCLUDE_PATH="$OPENFST_PREFIX/include${C_INCLUDE_PATH:+:$C_INCLUDE_PATH}" \
  LIBRARY_PATH="$OPENFST_PREFIX/lib${LIBRARY_PATH:+:$LIBRARY_PATH}" \
  CPPFLAGS="-I$OPENFST_PREFIX/include${CPPFLAGS:+ $CPPFLAGS}" \
  LDFLAGS="-L$OPENFST_PREFIX/lib${LDFLAGS:+ $LDFLAGS}" \
    "${installer[@]}" "$src_dir"

  vpy_has pynini || die "pynini 安装后仍无法导入"
fi

# ------------------------------------------------------------ 3. WeTextProcessing
# WeTextProcessing 的 METADATA 钉死 pynini==2.1.6（与 OpenFST 1.8 不兼容），
# 所以这里 --no-deps 安装，保留 pynini 2.1.7；实测 API 兼容。
if vpy_has tn; then
  log "WeTextProcessing 已安装"
else
  log "下载 WeTextProcessing ${WETEXT_VERSION}"
  simple_index="$PYPI_ROOT/simple/wetextprocessing/"
  href="$(curl -fsSL "$simple_index" \
    | grep -oiE "href=\"[^\"]*wetextprocessing-${WETEXT_VERSION}-py3-none-any\.whl[^\"]*\"" \
    | head -1 | sed -e 's/^href="//' -e 's/"$//' -e 's/#.*$//')"
  [[ -n "$href" ]] || die "在 $simple_index 找不到 WeTextProcessing ${WETEXT_VERSION}"

  case "$href" in
    http*) wheel_url="$href" ;;
    *) wheel_url="$PYPI_ROOT/${href#../../}" ;;
  esac
  log "wheel: $wheel_url"
  curl -fsSL -o "$WORK_DIR/WeTextProcessing-${WETEXT_VERSION}-py3-none-any.whl" "$wheel_url"
  "${installer[@]}" --no-deps "$WORK_DIR/WeTextProcessing-${WETEXT_VERSION}-py3-none-any.whl"
  "${installer[@]}" importlib-resources
fi

# ---------------------------------------------------------------- 4. soundfile
if vpy_has soundfile; then
  log "soundfile 已安装"
else
  log "安装 soundfile（torchaudio 在 macOS 缺 ffmpeg 后端）"
  "${installer[@]}" soundfile
fi

# ---------------------------------------------------------------- 5. 验证
log "验证安装结果"
"$VENV_PY" - <<'PY'
import pynini
from tn.chinese.normalizer import Normalizer as ZhNormalizer

print("pynini     :", pynini.__version__)
print("归一化示例 :", ZhNormalizer(overwrite_cache=False).normalize("日期是2026年10月22日。"))

try:
    import torchaudio
except ImportError:
    print("torchaudio : 未安装（先跑 uv pip install -r requirements.txt），跳过音频后端检查")
else:
    backends = torchaudio.list_audio_backends()
    print("torchaudio 后端 :", backends)
    assert "soundfile" in backends, "torchaudio 仍没有可用后端，读 wav 会失败"
print("OK")
PY

log "完成。可以运行: python infer.py --checkpoint MOSS-TTS-Nano-100M --prompt-audio-path assets/audio/zh_1.wav --text \"...\" --device cpu"
