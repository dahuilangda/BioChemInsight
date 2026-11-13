"""Dataset and image-only evaluation contracts for complete molecular recognition.

Gold annotations belong to the scorer, never the recognizer request. Oracle
experiments must use a separately named entry point and are not accepted here.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, Any
import hashlib


class ProtocolError(ValueError):
    pass


def canonical_smiles(text: str) -> str:
    """Preserve specified stereo, isotopes, charge and disconnected components."""
    from rdkit import Chem

    if not isinstance(text, str) or not text.strip():
        raise ProtocolError('Missing molecular target')
    options = Chem.SmilesParserParams()
    options.parseName = False
    options.allowCXSMILES = False
    molecule = Chem.MolFromSmiles(text, options)
    if molecule is None:
        raise ProtocolError('Invalid complete-molecule SMILES')
    if any(atom.GetAtomicNum() == 0 for atom in molecule.GetAtoms()):
        raise ProtocolError('Unresolved variable/dummy is outside complete-molecule scope')
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def _field(row: Mapping[str, Any], name: str, *aliases: str) -> Any:
    for key in (name, *aliases):
        if key in row:
            return row[key]
    raise ProtocolError(f'Missing {name}')


def assert_disjoint_splits(
    records: Iterable[Mapping[str, Any]], *,
    allowed_splits: tuple[str, ...] = ('train', 'validation', 'test'),
) -> dict[str, Any]:
    """Reject identity/component/image/family overlap, including reordered SMILES.

    ``component`` is mandatory: deriving a chemical identity alone cannot prove
    composition/patent-family separation. Optional family_id, composition_id and
    publication_family add stronger guards when their provenance exists.
    """
    owners: dict[tuple[str, str], str] = {}
    keys: set[str] = set()
    counts: Counter[str] = Counter()
    identities: dict[str, set[str]] = defaultdict(set)
    for row in records:
        key = _field(row, 'record_key')
        split = _field(row, 'split')
        component = _field(row, 'component')
        image_hash = _field(row, 'image_sha256')
        if not all(isinstance(value, str) and value for value in (key, split, component, image_hash)):
            raise ProtocolError('Record, split, component and image hash must be nonempty strings')
        if split not in allowed_splits:
            raise ProtocolError(f'Unknown split {split!r}')
        if key in keys:
            raise ProtocolError(f'Duplicate record_key: {key}')
        keys.add(key)
        identity = canonical_smiles(_field(row, 'smiles', 'SMILES'))
        guards = [('component', component), ('smiles', identity), ('image_sha256', image_hash)]
        for field in ('family_id', 'composition_id', 'publication_family'):
            if row.get(field) is not None:
                if not isinstance(row[field], str) or not row[field]:
                    raise ProtocolError(f'Invalid {field}')
                guards.append((field, row[field]))
        for field, value in guards:
            previous = owners.setdefault((field, value), split)
            if previous != split:
                raise ProtocolError(f'Cross-split {field} overlap: {previous} / {split}; record={key}')
        counts[split] += 1
        identities[split].add(identity)
    if not keys:
        raise ProtocolError('An empty dataset cannot establish a split contract')
    return {'records': len(keys), 'by_split': dict(counts),
            'unique_smiles': {split: len(values) for split, values in identities.items()},
            'cross_split_overlaps': 0,
            'historical_pretraining_overlap': 'unknown'}


@dataclass(frozen=True, slots=True)
class ImageInput:
    record_key: str
    image_path: str
    image_sha256: str
    input_size: int | None = None
    model_input_preprocessed: bool = False

    def verify_file(self) -> None:
        digest = hashlib.sha256(Path(self.image_path).read_bytes()).hexdigest()
        if digest != self.image_sha256:
            raise ProtocolError(f'Image bytes changed: {self.record_key}')

    def as_request(self) -> dict[str, Any]:
        request = asdict(self)
        assert_image_only_payload(request)
        return request


def image_only_input(record: Mapping[str, Any]) -> ImageInput:
    """Select only permitted image metadata from an annotated dataset record."""
    return ImageInput(
        record_key=_field(record, 'record_key'),
        image_path=_field(record, 'image_path', 'file_path'),
        image_sha256=_field(record, 'image_sha256'),
        input_size=record.get('input_size'),
        model_input_preprocessed=record.get('model_input_preprocessed', False),
    )


def assert_image_only_payload(payload: Mapping[str, Any]) -> None:
    allowed = {'record_key', 'image_path', 'image_sha256', 'input_size', 'model_input_preprocessed'}
    unknown = set(payload) - allowed
    if unknown:
        raise ProtocolError(f'Oracle/non-image fields in prediction request: {sorted(unknown)}')
    for name in ('record_key', 'image_path', 'image_sha256'):
        if not isinstance(payload.get(name), str) or not payload[name]:
            raise ProtocolError(f'Invalid {name}')
    if payload.get('input_size') is not None and (type(payload['input_size']) is not int or payload['input_size'] <= 0):
        raise ProtocolError('Invalid input_size')
    if type(payload.get('model_input_preprocessed', False)) is not bool:
        raise ProtocolError('Invalid preprocessing flag')


@dataclass(frozen=True, slots=True)
class EvaluationTarget:
    record_key: str
    smiles: str
    component: str = ''
    category: str = ''


@dataclass(frozen=True, slots=True)
class Prediction:
    record_key: str
    raw_smiles: str | None
    # A parseable string can still be a truncated, valid molecular fragment.
    # Only the decoder's observed EOS (or an explicit complete-graph protocol)
    # may set this true; construction alone must never imply termination.
    terminated: bool = False
    error: str | None = None
    repaired_smiles: str | None = None


def _compare(gold: str, prediction: str | None) -> dict[str, bool]:
    if prediction is None:
        return {'parse_success': False, 'literal_exact': False, 'resolved_exact': False}
    try:
        predicted = canonical_smiles(prediction)
    except ProtocolError:
        return {'parse_success': False, 'literal_exact': False, 'resolved_exact': False}
    exact = canonical_smiles(gold) == predicted
    return {'parse_success': True, 'literal_exact': exact, 'resolved_exact': exact}


def score_predictions(
    targets: Iterable[EvaluationTarget], predictions: Iterable[Prediction], *,
    compare_fn: Callable[[str, str | None], Mapping[str, bool]] | None = None,
) -> dict[str, Any]:
    """Score all frozen targets; missing/failed/unterminated results cannot vanish.

    The repaired pipeline passes unchanged raw outputs through when no repair
    was supplied; its results are always separate from raw recognition metrics.
    This function has no route for oracle atom counts/positions to reach a model.
    """
    target_rows = list(targets)
    target_keys = [row.record_key for row in target_rows]
    if not target_rows or len(set(target_keys)) != len(target_keys):
        raise ProtocolError('Targets must be nonempty with unique record keys')
    target_key_set = set(target_keys)
    for row in target_rows:
        canonical_smiles(row.smiles)
    indexed = {}
    for prediction in predictions:
        if prediction.record_key in indexed or prediction.record_key not in target_key_set:
            raise ProtocolError(f'Duplicate/unknown prediction: {prediction.record_key}')
        if type(prediction.terminated) is not bool:
            raise ProtocolError('Termination must be an explicit boolean')
        indexed[prediction.record_key] = prediction
    comparator = compare_fn or _compare
    rows = []
    counters = {name: Counter() for name in ('raw', 'repaired')}
    for target in target_rows:
        prediction = indexed.get(target.record_key)
        error = 'missing_prediction' if prediction is None else prediction.error
        terminated = prediction is not None and prediction.terminated
        repaired = prediction is not None and prediction.repaired_smiles is not None
        result = {'record_key': target.record_key, 'component': target.component,
                  'category': target.category, 'error': error, 'terminated': terminated,
                  'repair_applied': repaired}
        for name in ('raw', 'repaired'):
            text = None if prediction is None else prediction.raw_smiles
            if name == 'repaired' and repaired:
                text = prediction.repaired_smiles
            if error:
                metrics = {'parse_success': False, 'literal_exact': False, 'resolved_exact': False}
            else:
                metrics = dict(comparator(target.smiles, text))
                if any(type(metrics.get(key)) is not bool for key in ('parse_success', 'literal_exact', 'resolved_exact')):
                    raise ProtocolError('Comparator must report explicit boolean metrics')
                if not metrics['parse_success'] and (metrics['literal_exact'] or metrics['resolved_exact']):
                    raise ProtocolError('An unparsed prediction cannot be exact')
            if not terminated:
                metrics['literal_exact'] = metrics['resolved_exact'] = False
            result[name] = metrics
            counters[name].update({key: int(metrics[key]) for key in ('parse_success', 'literal_exact', 'resolved_exact')})
        rows.append(result)
    count = len(target_rows)
    return {'n': count, 'evaluation_mode': 'image_only',
            'failed': sum(bool(row['error']) for row in rows),
            'terminated': sum(row['terminated'] for row in rows),
            'repair_applied': sum(row['repair_applied'] for row in rows),
            **{name: {'denominator': count, **dict(values),
                      **{key + '_rate': values[key] / count for key in ('parse_success', 'literal_exact', 'resolved_exact')}}
               for name, values in counters.items()},
            'rows': rows}
