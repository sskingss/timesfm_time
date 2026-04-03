#!/usr/bin/env bash
# ─────────────────────────────────────────────────
# 每日信号扫描 — 由 launchd / cron 调用
# ─────────────────────────────────────────────────
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="$PROJECT_DIR/.venv/bin/python"
LOG_DIR="$PROJECT_DIR/logs"
LOG_FILE="$LOG_DIR/scan_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "$LOG_DIR"

echo "=== Daily Scan $(date) ===" | tee "$LOG_FILE"

cd "$PROJECT_DIR"
"$PYTHON" -m src.signal_runner --actionable-only 2>&1 | tee -a "$LOG_FILE"

# 清理 30 天前的日志
find "$LOG_DIR" -name "scan_*.log" -mtime +30 -delete 2>/dev/null || true

echo "=== Log saved: $LOG_FILE ===" | tee -a "$LOG_FILE"
