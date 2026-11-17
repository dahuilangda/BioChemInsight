import hashlib
import json
import subprocess
import unittest
from pathlib import Path


class ModelDeploymentTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[2]
        result = subprocess.run(
            ["docker", "compose", "config", "--format", "json"],
            cwd=cls.root,
            capture_output=True,
            text=True,
            check=True,
        )
        cls.services = json.loads(result.stdout)["services"]

    def test_services_share_model_and_sources(self):
        sources = set()
        for name in ("web", "worker", "harness-api"):
            service = self.services[name]
            environment = service["environment"]
            self.assertEqual(environment["MOLSIGHT_ARTIFACT"], "/models/molsight")
            self.assertEqual(
                environment["MOLSIGHT_CHECKPOINT"], "/models/molsight/model.pth"
            )
            mounts = {mount["target"]: mount for mount in service["volumes"]}
            self.assertEqual(
                Path(mounts["/app/training"]["source"]), self.root / "training"
            )
            self.assertTrue(mounts["/models/molsight"]["read_only"])
            self.assertTrue(mounts["/app/training"]["read_only"])
            sources.add(mounts["/models/molsight"]["source"])
        self.assertEqual(len(sources), 1)

    def test_artifact_matches_sources(self):
        mounts = {
            mount["target"]: mount for mount in self.services["harness-api"]["volumes"]
        }
        artifact = Path(mounts["/models/molsight"]["source"])
        if not (artifact / "artifact.json").exists():
            self.skipTest("Model release is not installed")
        metadata = json.loads((artifact / "artifact.json").read_text())
        for name, key in [
            ("model.pth", "model_sha256"),
            ("vocabulary.json", "vocabulary_sha256"),
        ]:
            with (artifact / name).open("rb") as stream:
                self.assertEqual(
                    hashlib.file_digest(stream, "sha256").hexdigest(), metadata[key]
                )
        for relative, digest in metadata["inference_sources"].items():
            self.assertEqual(
                hashlib.sha256((self.root / relative).read_bytes()).hexdigest(),
                digest,
                relative,
            )

    def test_runtime_mounts_exist(self):
        for name in ("web", "worker", "harness-api"):
            service = self.services[name]
            mounts = {
                mount["target"]: Path(mount["source"]) for mount in service["volumes"]
            }
            self.assertTrue((mounts["/opt/molsight"] / "bin/python").is_file())
            self.assertEqual(
                service["environment"]["MOLSIGHT_VENV_PYTHON"],
                "/opt/molsight/bin/python",
            )


if __name__ == "__main__":
    unittest.main()
