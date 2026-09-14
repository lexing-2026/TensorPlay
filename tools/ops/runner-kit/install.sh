#!/usr/bin/env bash
# Bootstrap one Linux self-hosted runner from the infra kit.
set -euo pipefail

KIT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOKEN="${1:?usage: ./install.sh <registration-token> [runner-name]}"
RNAME="${2:-tensorplay-runner-1}"
REPO_URL="https://github.com/lexing-2026/TensorPlay"
RELAY="${TP_RELAY:-/tmp/tp-relay}"
RUNNER_DIR="$RELAY/runner"
PROXY="http://127.0.0.1:7897"
NO_PROXY_DOMAINS="127.0.0.1,localhost,.p300s.com,10.0.0.0/8,.internal,.actions.githubusercontent.com,api.github.com"
TOOLKIT_ROOT="$(cd "$KIT/.." && pwd)/toolkits"

mkdir -p "$RELAY" "$HOME/.local/bin" "$HOME/.config/tensorplay"

if ! pgrep -f "mihomo -d" >/dev/null 2>&1; then
    mkdir -p "$RELAY/mihomo"
    if [ ! -f "$RELAY/mihomo/mihomo" ] && [ -d "$KIT/mihomo" ]; then
        cp -r "$KIT/mihomo/." "$RELAY/mihomo/"
    fi
    chmod +x "$RELAY/mihomo/mihomo"
    ( cd "$RELAY/mihomo" && nohup ./mihomo -d . -f config.yaml > mihomo.log 2>&1 & )
    sleep 4
fi
curl -s -o /dev/null --max-time 15 -x "$PROXY" https://github.com \
    || { echo "proxy self-check failed" >&2; tail -20 "$RELAY/mihomo/mihomo.log"; exit 1; }

cache_bin_tmp="$HOME/.local/bin/.sccache.$$"
cache_env_tmp="$HOME/.config/tensorplay/.s3-cache.env.$$"
install -m 0755 "$KIT/sccache-0.8.1-x86_64-unknown-linux-musl" "$cache_bin_tmp"
install -m 0600 "$KIT/s3-cache.env" "$cache_env_tmp"
mv -f "$cache_bin_tmp" "$HOME/.local/bin/sccache"
mv -f "$cache_env_tmp" "$HOME/.config/tensorplay/s3-cache.env"
grep -q ".local/bin" "$HOME/.bashrc" 2>/dev/null \
    || echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc"

if [ ! -d "$RELAY/mirrors" ]; then
    tar -C "$RELAY" -xzf "$KIT/mirrors.tar.gz"
fi
if ! grep -q "tp-mirror-rules" "$HOME/.gitconfig" 2>/dev/null; then
    {
        echo "# tp-mirror-rules (managed by install.sh)"
        sed -e "s#/home/bluemoon#$RELAY#g" -e "s#/actions-runner/mirrors#/mirrors#g" "$KIT/gitconfig-vendor.rules"
        printf '[url "file://%s/mirrors/TensorPlay.git"]\n\tinsteadOf = https://github.com/lexing-2026/TensorPlay\n' "$RELAY"
    } >> "$HOME/.gitconfig"
fi

if [ ! -f "$RUNNER_DIR/config.sh" ]; then
    mkdir -p "$RUNNER_DIR"
    tar -C "$RUNNER_DIR" -xzf "$KIT/actions-runner-linux-x64-2.337.0.tar.gz"
fi

CACHE_ENV="$(set -a; . "$HOME/.config/tensorplay/s3-cache.env"; echo "SCCACHE_BUCKET=$SCCACHE_BUCKET
SCCACHE_ENDPOINT=$SCCACHE_ENDPOINT
SCCACHE_REGION=${SCCACHE_REGION:-us-east-1}
SCCACHE_S3_USE_SSL=true
AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY
SCCACHE_NO_DAEMON=1
SCCACHE_IDLE_TIMEOUT=0
TP_S3_CACHE=1")"
{
    echo "HTTP_PROXY=$PROXY"
    echo "HTTPS_PROXY=$PROXY"
    echo "http_proxy=$PROXY"
    echo "https_proxy=$PROXY"
    echo "ALL_PROXY=$PROXY"
    echo "NO_PROXY=$NO_PROXY_DOMAINS"
    echo "no_proxy=$NO_PROXY_DOMAINS"
    echo "$CACHE_ENV"
    echo "TP_MKL_ROOT=$RELAY/mkl"
    echo "TP_TOOLKIT_ROOT=$TOOLKIT_ROOT"
} > "$RUNNER_DIR/.env"

cd "$RUNNER_DIR"
if [ -f .runner ]; then
    ./config.sh remove --token "$TOKEN" --name "$RNAME" >/dev/null 2>&1 || true
    sleep 2
fi
./config.sh --url "$REPO_URL" --token "$TOKEN" --name "$RNAME" \
    --labels "tensorplay-cuda,$RNAME" --work "$RUNNER_DIR/_work" --unattended

if ! pgrep -f "$RUNNER_DIR/bin/Runner.Listener" >/dev/null 2>&1; then
    nohup bash -lc "cd '$RUNNER_DIR' && exec ./run.sh" > "$RUNNER_DIR/run.log" 2>&1 &
    sleep 6
fi
tail -3 "$RUNNER_DIR/run.log" 2>/dev/null || true
echo "[ok] runner '$RNAME' configured with S3 cache"
