from PIL import Image, ImageDraw
import pytest

from benchmark import RecognitionImages, compare, digest


def test_chemical_interpretation_and_literal_transcription_are_separate_metrics():
    assert compare("Nc1ccccc1", "N*<sep><a>1:Ph</a>") == {
        "parse_success": True,
        "literal_exact": False,
        "resolved_exact": True,
    }
    assert not compare("Nc1ccccc1", "N*<sep><a>1:Ar</a>")["resolved_exact"]
    assert not compare("N[C@@H](C)CO", "N[C@H](C)CO")["resolved_exact"]
    assert not compare("C[*:1]", "C[1*]")["resolved_exact"]
    assert not compare("CC", "invalid")["parse_success"]


def test_verified_images_keep_their_validated_frame(tmp_path):
    path = tmp_path / "input.png"
    image = Image.new("RGB", (512, 512), "white")
    ImageDraw.Draw(image).rectangle((200, 200, 300, 300), fill="black")
    image.save(path)
    pairs = [
        {
            "file_path": str(path),
            "model_input_preprocessed": True,
            "image_sha256": digest(path),
        }
    ]
    index, tensor = RecognitionImages(pairs, 512)[0]
    assert index == 0
    assert tensor.shape == (3, 512, 512)
    assert (tensor[0] < 0).nonzero().amin(0).tolist() == [200, 200]
    assert (tensor[0] < 0).nonzero().amax(0).tolist() == [300, 300]
    with pytest.raises(ValueError, match="cannot be cropped or downsampled"):
        RecognitionImages(pairs, 256)[0]
    image.putpixel((0, 0), (0, 0, 0))
    image.save(path)
    with pytest.raises(ValueError, match="changed after validation"):
        RecognitionImages(pairs, 512)[0]
