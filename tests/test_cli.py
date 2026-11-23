from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from rag_eval.cli import app

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "providers.yaml"
runner = CliRunner()


@pytest.fixture(autouse=True)
def _isolated_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Run from a temp dir so a developer's real .env is never picked up.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RAG_EVAL_PROVIDERS_CONFIG", str(CONFIG_PATH))


def test_doctor_fails_when_no_llm_provider_is_usable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acct123")
    result = runner.invoke(app, ["doctor"])

    assert result.exit_code == 1
    assert "missing credentials" in result.output


def test_doctor_passes_with_a_single_configured_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acct123")
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test")
    result = runner.invoke(app, ["doctor"])

    assert result.exit_code == 0, result.output
    assert "nvidia_super" in result.output
    assert "ready" in result.output


def test_doctor_strict_fails_on_partial_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acct123")
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test")
    result = runner.invoke(app, ["doctor", "--strict"])

    assert result.exit_code == 1
    assert "strict" in result.output


def test_doctor_reports_unresolved_cloudflare_account_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CLOUDFLARE_ACCOUNT_ID", raising=False)
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test")
    result = runner.invoke(app, ["doctor"])

    assert result.exit_code == 0, result.output
    assert "CLOUDFLARE_ACCOUNT_ID" in result.output


def test_doctor_never_prints_key_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acct123")
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-super-secret-value")
    result = runner.invoke(app, ["doctor"])

    assert "nvapi-super-secret-value" not in result.output


def test_doctor_rejects_unknown_live_target(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acct123")
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test")
    result = runner.invoke(app, ["doctor", "--live", "--target", "banana"])

    assert result.exit_code == 2


def test_cache_stats_requires_an_existing_cache() -> None:
    result = runner.invoke(app, ["cache-stats"])
    assert result.exit_code == 1
    assert "cache not found" in result.output
