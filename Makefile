# ================================================================
#  TimesFM 量化交易系统 — Makefile
# ================================================================
#  回测:  make backtest
#  预测:  make scan / make scan-buy-sell
# ================================================================

PYTHON := uv run python

.DEFAULT_GOAL := help

# ── 回测 ──────────────────────────────────────────────────────────

.PHONY: backtest
backtest: ## 执行完整回测（全部标的 × 全部策略）
	$(PYTHON) -m src.main

# ── 实时预测 / 信号扫描 ──────────────────────────────────────────

.PHONY: scan
scan: ## 扫描最新信号（全部标的 × 全部策略，含 HOLD）
	$(PYTHON) -m src.signal_runner

.PHONY: scan-buy-sell
scan-buy-sell: ## 只推送买入/卖出信号，过滤 HOLD
	$(PYTHON) -m src.signal_runner --actionable-only

.PHONY: scan-symbols
scan-symbols: ## 指定标的扫描, 用法: make scan-symbols SYMBOLS="601766 600519"
	$(PYTHON) -m src.signal_runner --symbols $(SYMBOLS)

.PHONY: scan-strategy
scan-strategy: ## 指定策略扫描, 用法: make scan-strategy STRATEGY="分位数"
	$(PYTHON) -m src.signal_runner --strategy "$(STRATEGY)"

# ── 市场分析报告 ──────────────────────────────────────────────────

.PHONY: report
report: ## 生成A股短线市场分析报告（风格趋势+资金流向+个股精选）
	$(PYTHON) -m src.market_report

.PHONY: report-top
report-top: ## 指定精选个股数量, 用法: make report-top TOP=15
	$(PYTHON) -m src.market_report --top $(TOP)

# ── 定时任务 ──────────────────────────────────────────────────────

.PHONY: daily
daily: ## 执行每日定时扫描脚本（含日志记录）
	bash scripts/daily_scan.sh

# ── 工具 ──────────────────────────────────────────────────────────

.PHONY: clean
clean: ## 清理缓存和输出文件
	rm -rf cache/market_data/*.pkl output/*.png logs/scan_*.log

.PHONY: help
help: ## 显示帮助信息
	@echo ""
	@echo "  TimesFM 量化交易系统"
	@echo "  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo ""
	@echo "  回测:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | grep -E 'backtest' | awk 'BEGIN {FS = ":.*?## "}; {printf "    \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "  预测/扫描:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | grep -E 'scan|daily' | awk 'BEGIN {FS = ":.*?## "}; {printf "    \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "  市场报告:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | grep -E 'report' | awk 'BEGIN {FS = ":.*?## "}; {printf "    \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "  工具:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | grep -E 'clean|help' | awk 'BEGIN {FS = ":.*?## "}; {printf "    \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""
