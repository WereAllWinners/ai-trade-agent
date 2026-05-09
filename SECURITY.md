# Security Audit — ai-trade-agent

**Audited:** 2026-05-08  
**Remediated:** 2026-05-08  
**Tool:** pip-audit (against project venv at `venv/`)  
**Packages scanned:** 226  

---

## Secret / Credential Hygiene

| Check | Result |
|---|---|
| `.env` in `.gitignore` | ✅ Present (exact match) |
| `*.env` in `.gitignore` | ✅ Present (wildcard match) |
| `.env` ever committed to git history | ✅ Never committed |
| `.env.example` committed (safe placeholder values) | ✅ Correct |

No live API keys or secrets have ever been pushed to the repository.

---

## Vulnerability Summary

**18 packages flagged** across the full venv. Findings are grouped by runtime relevance.

### Priority 1 — Trading Runtime (address promptly)

| Package | Version | CVEs | Risk | Status | Notes |
|---|---|---|---|---|---|
| `aiohttp` | ~~3.13.3~~ → **3.13.5** | CVE-2026-34513 through -34525, CVE-2026-22815 | Medium | ✅ **Fixed** | 10 CVEs: unbounded DNS cache, multipart DoS, header injection, redirect cookie leak. Used by alpaca-py async client. |
| `transformers` | ~~4.57.6~~ → **5.5.0** | CVE-2026-1839 | Medium | ✅ **Fixed** | RCE via `_load_rng_state()` in `Trainer`. Only reachable during fine-tuning. Unsloth also upgraded to 2026.5.2 for 5.x compatibility. |
| `requests` | 2.32.5 | CVE-2026-25645 | Low | Accepted | Predictable temp file during zip extraction. No attacker-controlled archive input in trading flows. |
| `setuptools` | 70.2.0 | PYSEC-2025-49 | Low | Accepted | Path traversal in `PackageIndex._download_url()`. Install-time only, not runtime. |
| `python-dotenv` | 1.2.1 | CVE-2026-28684 | Low | Accepted | Symlink follow in `set_key()` / `unset_key()`. Not called — `.env` is read-only at startup. |

**Completed upgrades:**
```bash
pip install "aiohttp>=3.13.5" --upgrade   # ✅ done 2026-05-08
pip install "transformers>=5.0.0" --upgrade  # ✅ done 2026-05-08 (resolved to 5.5.0 via unsloth)
pip install "unsloth>=2026.5.0" --upgrade    # ✅ done 2026-05-08 (2026.5.2, supports transformers 5.x)
```

### Priority 2 — Development / Notebook Only

These packages are not imported by the trading daemons or agents. They are present because the venv also serves Jupyter notebooks used for analysis.

| Package | CVEs | Notes |
|---|---|---|
| `jupyter-server` 2.17.0 | 4 CVEs | Open redirect, origin validation bypass, path traversal, cookie secret. Not exposed in production — notebook server runs locally only. |
| `jupyterlab` 4.5.4 | 2 CVEs | Extension allowlist bypass, XSS via `data-commandlinker`. Local notebook use only. |
| `notebook` 7.5.3 | CVE-2026-40171 | Stored XSS. Local use only. |
| `nbconvert` 7.17.0 | 2 CVEs | Arbitrary file read/write via path traversal in notebook exports. |
| `pillow` 12.1.1 | 4 CVEs | FITS/PDF/PSD parsing flaws. Not used in trading path. |
| `mistune` 3.2.0 | 3 CVEs | Markdown ReDoS and HTML injection. Jupyter dependency. |
| `tornado` 6.5.4 | 3 CVEs | Cookie attribute injection, multipart DoS. Jupyter dependency. |
| `lxml` 6.0.2 | CVE-2026-41066 | XXE file read with default config. Not used in trading path. |
| `curl-cffi` 0.13.0 | CVE-2026-33752 | SSRF via redirect following. Not used in trading path. |
| `diffusers` 0.36.0 | CVE-2026-44513 | `trust_remote_code` bypass. Not used in this project. |
| `pygments` 2.19.2 | CVE-2026-4539 | ReDoS in ADL lexer. Dev tooling only. |
| `pytest` 9.0.2 | CVE-2025-71176 | Temp dir privilege escalation on UNIX. CI/test only. |
| `pip` 26.0.1 | 2 CVEs | Zip/tar handling, deferred import. Upgrade via `pip install --upgrade pip`. |

**Recommended upgrades (dev environment):**
```bash
pip install --upgrade jupyter-server jupyterlab notebook nbconvert pillow pip
```

---

## Actions Taken

- Verified `.env` was never committed to git history (`git log --all --full-history -- .env*`)
- Confirmed `.env` and `*.env` are in `.gitignore`
- Upgraded `aiohttp` 3.13.3 → 3.13.5 (clears all 10 CVEs)
- Upgraded `transformers` 4.57.6 → 5.5.0 + `unsloth` 2026.2.1 → 2026.5.2 (clears CVE-2026-1839)
- Pinned new minimum versions in `requirements.txt`
- Re-ran pip-audit post-upgrade: `aiohttp` and `transformers` both show **CLEAN**
- Re-ran full test suite post-upgrade: **255/255 passed**

---

## Ongoing Recommendations

1. Run `pip-audit` monthly or add to CI pipeline (`pip-audit --format cyclonedx-json -o sbom.json`).
2. Keep `aiohttp` pinned to latest in `requirements.txt` — most CVEs are Alpaca client surface.
3. Consider separating trading venv from notebook/analysis venv to reduce audit noise.
4. Rotate Alpaca API keys periodically regardless of secret hygiene — Alpaca paper keys expire on inactivity.
