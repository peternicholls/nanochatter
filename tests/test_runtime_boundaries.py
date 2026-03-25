import argparse
import logging

import nanochat.common as common


def test_chat_web_import_has_no_cli_or_compute_side_effects(monkeypatch, fresh_import):
    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", lambda self, *args, **kwargs: (_ for _ in ()).throw(AssertionError("parse_args called during import")))
    monkeypatch.setattr(common, "compute_init", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("compute_init called during import")))

    module = fresh_import("scripts.chat_web")

    assert module.app is not None
    assert module.app.state.settings.source == "sft"


def test_tokenizer_import_does_not_require_training_only_rustbpe(blocked_import, fresh_import):
    with blocked_import("rustbpe"):
        module = fresh_import("nanochat.tokenizer")

    assert hasattr(module, "RustBPETokenizer")
    assert callable(module.get_tokenizer)


def test_checkpoint_manager_import_does_not_require_training_only_rustbpe_or_reconfigure_logging(
    monkeypatch,
    blocked_import,
    fresh_import,
):
    monkeypatch.setattr(common, "setup_default_logging", lambda: (_ for _ in ()).throw(AssertionError("setup_default_logging called during import")))
    monkeypatch.setattr(logging, "basicConfig", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("logging.basicConfig called during import")))

    with blocked_import("rustbpe"):
        module = fresh_import("nanochat.checkpoint_manager")

    assert hasattr(module, "load_model")