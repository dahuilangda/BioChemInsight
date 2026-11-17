# Model deployment

The inference services read the model release from `/models/molsight`. The directory contains `model.pth`, `artifact.json` and `vocabulary.json`. Configure its host location with `MOLSIGHT_MODEL_DIR` in `.env`.

Store releases under `models/molsight/<sha256-prefix>/`, using the checkpoint's SHA-256 digest. Keep each release immutable. The current release is `491c648a203c`, with checkpoint digest `491c648a203c6ebdf90911dfe0716effbcf5bd1796a985ef28391351d432041e`.

All three inference consumers mount the repository's `training` directory at `/app/training`. The artifact records the required inference source digests; the worker checks these, the vocabulary and checkpoint before loading. Do not replace the training directory with an unrelated Git worktree.

The local Compose override runs the inference interpreter at `/opt/molsight/bin/python`. `MOLSIGHT_RUNTIME_DIR` identifies that virtual environment on the host. `MOLSIGHT_PYTHON_HOME` identifies its base Python installation and is mounted at its original absolute path because virtual environment executables may reference it. These host locations belong in `.env`, not model code. The standalone harness image uses its own `/opt/venv` interpreter.

`docker-compose.override.yml` and `.env` are local deployment configuration. The base Compose file and `Dockerfile.harness` use the same model directory and worker. The image copies only the inference source packages; weights are mounted separately.

Validation:

```sh
docker compose config --quiet
python -m unittest discover -s frontend/tests -p deployment_test.py
```

The API response's `model` field reports the actual model digest, artifact path, recognition branch and input digest. `/healthz` reports process readiness and configured checkpoint location; it does not establish recognition accuracy.

Historical experiment reports retain their original artifact names and paths for traceability. They are not deployment settings.
