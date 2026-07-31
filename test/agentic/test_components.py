from __future__ import annotations

import pytest

from lmms_eval.agentic import (
    FixedActionModelServer,
    IdentityModelOutputParser,
    ModelServer,
    QwenModelOutputParser,
)
from lmms_eval.agentic.components import (
    call_with_accepted_kwargs,
    import_from_path,
    resolve,
)
from lmms_eval.agentic.parsers import ModelOutputParser


def test_resolve_returns_instances_unchanged():
    server = FixedActionModelServer(action="X")
    assert resolve("model_server", server, expected=ModelServer) is server


def test_resolve_registry_name():
    server = resolve("model_server", "debug", expected=ModelServer)
    assert isinstance(server, FixedActionModelServer)


def test_resolve_dict_spec_passes_kwargs():
    server = resolve("model_server", {"name": "debug", "action": "MOVE_LEFT"}, expected=ModelServer)
    assert server.action == "MOVE_LEFT"


def test_resolve_import_path():
    parser = resolve("model_output_parser", "lmms_eval.agentic.parsers:QwenModelOutputParser", expected=ModelOutputParser)
    assert isinstance(parser, QwenModelOutputParser)


def test_resolve_callable_factory_with_context_filtering():
    seen = {}

    def factory(doc):
        seen["doc"] = doc
        return IdentityModelOutputParser()

    resolve("model_output_parser", factory, expected=ModelOutputParser, doc={"id": 1}, lmms_eval_specific_kwargs={"unused": True})
    assert seen["doc"] == {"id": 1}


def test_resolve_dict_kwargs_win_over_context():
    def factory(action="default"):
        return FixedActionModelServer(action=action)

    server = resolve("model_server", {"factory": factory, "action": "explicit"}, expected=ModelServer, action="context")
    assert server.action == "explicit"


def test_resolve_unknown_name_lists_builtins():
    with pytest.raises(KeyError, match="debug"):
        resolve("model_server", "nope", expected=ModelServer)


def test_resolve_wrong_return_type_raises():
    with pytest.raises(TypeError, match="expected ModelServer"):
        resolve("model_server", lambda: IdentityModelOutputParser(), expected=ModelServer)


def test_resolve_none_spec_raises():
    with pytest.raises(TypeError, match="required"):
        resolve("env_manager", None, expected=object)


def test_import_from_path_colon_and_dotted():
    assert import_from_path("lmms_eval.agentic.servers:FixedActionModelServer") is FixedActionModelServer
    assert import_from_path("lmms_eval.agentic.servers.FixedActionModelServer") is FixedActionModelServer


def test_call_with_accepted_kwargs_filters_by_signature():
    def strict(a):
        return a

    assert call_with_accepted_kwargs(strict, {"a": 1, "b": 2}) == 1

    def open_kwargs(**kwargs):
        return kwargs

    assert call_with_accepted_kwargs(open_kwargs, {"a": 1, "b": 2}) == {"a": 1, "b": 2}
