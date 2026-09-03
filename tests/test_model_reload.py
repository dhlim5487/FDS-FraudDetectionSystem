"""
tests/test_model_reload.py - the serving side of the retraining loop.

offline/promote.py replaces data/model.txt when a candidate clears the gate.
serving/app.py has to notice. It checks the file's mtime on every request and
reloads only when that changed, so this test pins both halves of that:
reloading when the file moved, and NOT reloading when it did not. Without the
second half, "read the whole model on every request" would also pass.

Redis is never contacted here - redis.Redis() connects lazily, so importing
serving.app is safe with nothing running.
"""
import os
import shutil

import pytest

PRODUCTION_MODEL = "data/model.txt"


@pytest.fixture
def app_with_temp_model(tmp_path, monkeypatch):
    """serving.app pointed at a throwaway copy of the model."""
    if not os.path.exists(PRODUCTION_MODEL):
        pytest.skip("no trained model - run offline.train first")

    import serving.app as app

    model_file = tmp_path / "model.txt"
    shutil.copy2(PRODUCTION_MODEL, model_file)

    monkeypatch.setattr(app, "MODEL_FILE", str(model_file))
    monkeypatch.setattr(app, "model_mtime", os.path.getmtime(model_file))
    return app, model_file


def test_no_reload_while_the_file_sits_still(app_with_temp_model):
    app, _ = app_with_temp_model

    first = app.current_model()
    assert app.current_model() is first, "reloaded a file that never changed"


def test_reloads_after_the_file_is_replaced(app_with_temp_model):
    app, model_file = app_with_temp_model

    first = app.current_model()
    os.utime(model_file, (0, 0))  # what promote.py's rename looks like from here

    assert app.current_model() is not first, "kept serving the old model"


def test_promote_swaps_atomically(tmp_path):
    """A reader must never catch install() mid-copy: rename, not overwrite."""
    from offline.promote import install

    src = tmp_path / "candidate.txt"
    dst = tmp_path / "model.txt"
    src.write_text("new")
    dst.write_text("old")

    seen = []
    real_copy = shutil.copy2

    def spy(a, b, *args, **kwargs):
        # mid-install: whatever a reader opens now must still be the old file
        seen.append(dst.read_text())
        return real_copy(a, b, *args, **kwargs)

    import offline.promote as promote

    original = promote.shutil.copy2
    promote.shutil.copy2 = spy
    try:
        install(src, dst)
    finally:
        promote.shutil.copy2 = original

    assert seen == ["old"], "target was already disturbed before the swap"
    assert dst.read_text() == "new"
    assert not (tmp_path / "model.txt.tmp").exists(), "temp file left behind"
