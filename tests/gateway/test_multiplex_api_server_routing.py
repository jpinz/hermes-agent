"""Multiplex /p/<profile>/ routing for the api_server adapter.

Mirrors ``test_multiplex_http_routing.py`` (webhook): the default listener
owns the port, and secondary profiles are reached via a URL prefix when
``gateway.multiplex_profiles`` is on.
"""
from __future__ import annotations

import json
import shlex
import sys
from pathlib import Path
from typing import Any, cast

import pytest
from aiohttp import FormData, web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import GatewayConfig, PlatformConfig
from gateway.platforms.api_server import (
    APIServerAdapter,
    _PROFILE_REJECTED,
    _api_request_profile,
)


def _make_adapter(
    multiplex: bool = True, allowlist: list[str] | None = None
) -> APIServerAdapter:
    cfg = PlatformConfig(enabled=True, extra={"host": "127.0.0.1", "port": 8642, "key": "test-key"})
    adapter = APIServerAdapter(cfg)

    class _Runner:
        config = GatewayConfig(
            multiplex_profiles=multiplex,
            multiplex_profile_allowlist=allowlist,
        )

    adapter.gateway_runner = _Runner()
    return adapter


class _FakeReq:
    def __init__(self, profile=None):
        self.match_info = {"profile": profile} if profile is not None else {}


def _audio_form(**fields: str) -> FormData:
    form = FormData()
    form.add_field("file", b"fake audio", filename="clip.wav", content_type="audio/wav")
    for name, value in fields.items():
        form.add_field(name, value)
    return form


def _write_command_stt_config(home: Path, transcript: str) -> None:
    home.mkdir(parents=True, exist_ok=True)
    code = (
        "import pathlib,sys; "
        "pathlib.Path(sys.argv[1]).write_text(sys.argv[2], encoding='utf-8')"
    )
    command = (
        f"{shlex.quote(sys.executable)} -c "
        f"{shlex.quote(code)} {{output_path}} {shlex.quote(transcript)}"
    )
    (home / "config.yaml").write_text(
        "stt:\n"
        "  enabled: true\n"
        "  provider: routecheck\n"
        "  providers:\n"
        "    routecheck:\n"
        "      type: command\n"
        f"      command: {json.dumps(command)}\n",
        encoding="utf-8",
    )


class TestApiServerProfileResolution:
    def test_no_prefix_returns_none(self):
        adapter = _make_adapter(multiplex=True)
        assert adapter._resolve_request_profile(_FakeReq(None)) is None

    def test_unserved_prefix_is_rejected(self, monkeypatch):
        adapter = _make_adapter(multiplex=True, allowlist=["worker"])
        monkeypatch.setattr(
            "hermes_cli.profiles.profiles_to_serve",
            lambda multiplex, profile_allowlist=None: [
                ("default", "/profiles/default"),
                ("worker", "/profiles/worker"),
            ],
        )

        assert (
            adapter._resolve_request_profile(cast(Any, _FakeReq("worker")))
            == "worker"
        )
        assert (
            adapter._resolve_request_profile(cast(Any, _FakeReq("restricted")))
            is _PROFILE_REJECTED
        )


class TestApiServerRouteTable:
    def test_route_table_includes_models_options_and_chat(self):
        """Model discovery and chat routes must survive profile multiplexing."""
        adapter = _make_adapter(multiplex=True)
        paths = {path for _method, path, _handler in adapter._http_route_table()}
        assert "/v1/models" in paths
        assert "/api/model/options" in paths
        assert "/v1/chat/completions" in paths
        assert "/v1/audio/transcriptions" in paths
        assert "/api/sessions/{session_id}/model" in paths
        # connect() mirrors every native path under /p/{profile}/…
        mirrored = {f"/p/{{profile}}{path}" for path in paths}
        assert "/p/{profile}/v1/models" in mirrored
        assert "/p/{profile}/api/model/options" in mirrored
        assert "/p/{profile}/v1/chat/completions" in mirrored
        assert "/p/{profile}/v1/audio/transcriptions" in mirrored
        assert "/p/{profile}/api/sessions/{session_id}/model" in mirrored


class TestApiServerModelsUnderProfile:
    def test_resolve_model_name_follows_active_profile(self, monkeypatch):
        """When the request is scoped to a named profile, advertise that name."""
        adapter = _make_adapter(multiplex=True)
        adapter._model_name = "hermes-agent"
        monkeypatch.setattr(
            "hermes_cli.profiles.get_active_profile_name",
            lambda: "coder",
        )
        token_prof = _api_request_profile.set("coder")
        try:
            assert adapter._resolve_model_name("") == "coder"
        finally:
            _api_request_profile.reset(token_prof)


class TestApiServerAudioTranscriptionsUnderProfile:
    @pytest.mark.asyncio
    async def test_prefixed_route_uses_profile_stt_config_in_worker_thread(
        self, tmp_path, monkeypatch
    ):
        default_home = tmp_path / ".hermes"
        profile_home = default_home / "profiles" / "worker"
        _write_command_stt_config(default_home, "default transcript")
        _write_command_stt_config(profile_home, "profile transcript")
        (profile_home / ".env").write_text(
            "API_SERVER_KEY=profile-test-key-000000\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("HERMES_HOME", str(default_home))
        monkeypatch.setattr(
            "hermes_cli.profiles.profiles_to_serve",
            lambda multiplex, profile_allowlist=None: [
                ("default", str(default_home)),
                ("worker", str(profile_home)),
            ],
        )
        monkeypatch.setattr(
            "hermes_cli.profiles.get_profile_dir",
            lambda profile: profile_home if profile == "worker" else default_home,
        )

        adapter = _make_adapter(multiplex=True, allowlist=["worker"])
        app = web.Application(middlewares=[adapter._make_profile_prefix_middleware()])
        app.router.add_post(
            "/p/{profile}/v1/audio/transcriptions",
            adapter._handle_audio_transcriptions,
        )

        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/p/worker/v1/audio/transcriptions",
                headers={
                    "Authorization": "Bearer " + "profile-test-key-000000"
                },
                data=_audio_form(model="whisper-1"),
            )
            body = await resp.json()

        assert resp.status == 200
        assert body == {"text": "profile transcript"}
