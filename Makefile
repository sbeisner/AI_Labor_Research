.PHONY: compute render clean-cache notebook help \
        fig-sarima-stationarity fig-sarima-aic fig-sarima-diagnostics \
        fig-control-vs-sarima fig-ai-vs-control fig-abm-churn \
        fig-beveridge fig-ks-test fig-km-survival fig-gsa

# ── Full workflow ─────────────────────────────────────────────────────────────

# Pre-execute all Python blocks and populate the Jupyter cache.
# Run after changing simulation code or data (slow, one-time).
compute:
	@echo "Pre-computing all Python blocks..."
	@rm -rf _cache/
	quarto render manuscript.qmd
	@echo "Done. Run 'make render' for fast PDF rebuilds."

# Render PDF using cached outputs — fast, no Python re-execution.
# (Only re-executes cells whose code has changed since last compute.)
render:
	quarto render manuscript.qmd

# Open manuscript as a Jupyter notebook for interactive cell-by-cell execution.
notebook:
	jupyter lab manuscript.quarto_ipynb

# Wipe the Jupyter cache so next render re-executes everything.
clean-cache:
	rm -rf _cache/

# ── Per-figure preview (standalone) ──────────────────────────────────────────
# Run any of these to regenerate a single figure and save to output/preview/.
# Useful when iterating on a plot without waiting for a full render.

fig-sarima-stationarity:
	python scripts/sarima_stationarity.py

fig-sarima-aic:
	python scripts/sarima_aic_selection.py

fig-sarima-diagnostics:
	python scripts/sarima_diagnostics.py

fig-control-vs-sarima:
	python scripts/control_vs_sarima_plot.py

fig-ai-vs-control:
	python scripts/ai_vs_control_plot.py

fig-abm-churn:
	python scripts/abm_churn_plot.py

fig-beveridge:
	python scripts/validation_beveridge.py

fig-ks-test:
	python scripts/validation_ks.py

fig-km-survival:
	python scripts/validation_km.py

fig-gsa:
	python scripts/validation_gsa_plot.py

help:
	@echo "Full workflow:"
	@echo "  make compute      - Re-execute all Python blocks and cache (slow)"
	@echo "  make render       - Render PDF using cached outputs (fast)"
	@echo "  make notebook     - Open in Jupyter Lab for interactive editing"
	@echo "  make clean-cache  - Wipe cache"
	@echo ""
	@echo "Per-figure preview (saves PNG to output/preview/):"
	@echo "  make fig-sarima-stationarity"
	@echo "  make fig-sarima-aic"
	@echo "  make fig-sarima-diagnostics"
	@echo "  make fig-control-vs-sarima"
	@echo "  make fig-ai-vs-control"
	@echo "  make fig-abm-churn"
	@echo "  make fig-beveridge"
	@echo "  make fig-ks-test"
	@echo "  make fig-km-survival"
	@echo "  make fig-gsa"
