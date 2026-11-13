"""Keep local OPSIN calls independent of process cwd, UID and concurrent calls."""
from pathlib import Path
import tempfile


def name_to_smiles(name, **options):
    from py2opsin import py2opsin
    if not isinstance(name, str) or not name.strip() or len(name) > 4096:
        return ''
    # py2opsin's default fixed cwd filename races across threads and cannot be
    # written by the production service user in a read-only application tree.
    with tempfile.TemporaryDirectory(prefix='bci-opsin-') as temporary:
        return py2opsin(name, tmp_fpath=str(Path(temporary) / 'input.txt'), **options)
