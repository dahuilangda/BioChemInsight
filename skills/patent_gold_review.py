"""Evidence contract for patent image/SMILES training references.

Mass agreement and a neighbouring OCR sequence label do not establish a pair.
Only an explicitly reviewed structure bound to the exact crop is accepted.
"""
import hashlib
from pathlib import PurePosixPath, Path


def validate_reference(record, crop=None, seq_id=None):
    if not isinstance(record, dict):
        return False, 'legacy_mass_only_reference'
    review = record.get('review', {})
    if review.get('status') != 'figure_reviewed':
        return False, 'reference_not_figure_reviewed'
    smiles = record.get('smiles')
    if not isinstance(smiles, str) or not smiles:
        return False, 'reference_missing_smiles'
    digest = hashlib.sha256(smiles.encode()).hexdigest()
    if review.get('smiles_sha256') != digest:
        return False, 'reference_changed_since_review'
    if seq_id is not None and str(record.get('seq_id')) != str(seq_id):
        return False, 'review_sequence_mismatch'
    if crop is not None:
        # Match patent + full crop identity, not a numeric seq label alone.
        actual = PurePosixPath(str(crop).replace('\\', '/'))
        expected = PurePosixPath(str(review.get('crop', '')))
        if len(expected.parts) < 2 or tuple(actual.parts[-len(expected.parts):]) != expected.parts:
            return False, 'review_crop_mismatch'
        expected_hash = review.get('crop_sha256')
        if not expected_hash:
            return False, 'reference_missing_crop_digest'
        image_path = Path(crop)
        if not image_path.is_absolute():
            image_path = Path(__file__).resolve().parents[1]/'data'/'real_crops'/image_path
        if not image_path.is_file():
            return False, 'review_crop_file_missing'
        if hashlib.sha256(image_path.read_bytes()).hexdigest() != expected_hash:
            return False, 'review_crop_changed_since_review'
    if not review.get('source_pdf_sha256') or not review.get('checks'):
        return False, 'reference_missing_source_evidence'
    return True, ''
