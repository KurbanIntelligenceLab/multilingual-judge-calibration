"""Build the supplementary saved-score archive outside the public checkout."""
import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
FILES = [
    'LICENSE', 'requirements.txt',
    'docs/REPRODUCE.md', 'docs/AI_ASSISTANCE.md',
    'paper/figures/fig_overview.tex',
    'scripts/generate_aamas_figures.py',
    'scripts/analyze_replication.py', 'scripts/analyze_panel_sensitivity.py',
    'scripts/audit_localization_and_transforms.py',
    'scripts/analyze_mrewardbench_external_pilot.py',
    'scripts/compute_meb_foundations.py',
    'scripts/generate_system_prompt_semantic_similarity.py',
    'agent_as_a_judge/languages.py',
]
FILES += [path.relative_to(ROOT).as_posix()
          for path in (ROOT / 'paper/code/analysis').glob('*.py')]
FILES += [path.relative_to(ROOT).as_posix()
          for path in (ROOT / 'agent_as_a_judge/module/prompt').glob('*.py')]
FILES += [path.relative_to(ROOT).as_posix()
          for pattern in ('*.csv', '*.json')
          for path in (ROOT / 'paper/analysis').glob(pattern)
          if path.name not in {
              'cbc_status.json', 'information_theoretic_measurement.json',
              'requirement_type_cbc.json',
          }]
FILES += [path.relative_to(ROOT).as_posix()
          for pattern in ('*.csv', '*.json', '*.jsonl')
          for path in (ROOT / 'paper/analysis/aamas_2027').rglob(pattern)
          if not path.name.endswith('_rerun.json')]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True,
                        help='ZIP destination outside this repository')
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    if output.is_relative_to(ROOT.resolve()):
        parser.error('Choose an output path outside the public repository')
    if output.suffix.lower() != '.zip':
        parser.error('The output filename must end in .zip')

    # The template lives under docs/ in the checkout, but becomes README.md
    # at the archive root. Adjust only its two documentation links.
    readme = (ROOT / 'docs/README_SUPPLEMENT.md').read_text(encoding='utf-8')
    for name in ('REPRODUCE.md', 'AI_ASSISTANCE.md'):
        readme = readme.replace(f']({name})', f'](docs/{name})')
    payload = {'README.md': readme.encode('utf-8')}
    for name in sorted(set(FILES)):
        path = ROOT / name
        if not path.is_file():
            raise FileNotFoundError(path)
        payload[name] = path.read_bytes()

    profile_roots = (b'C:' + bytes([92]) + b'Users' + bytes([92]), b'C:/Users/')
    for name, data in payload.items():
        if any(marker in data for marker in profile_roots):
            raise ValueError(f'Private filesystem path in {name}')
    manifest = {
        'schema_version': 1,
        'paper_title': 'Language-Conditioned Rank Reversal in Agentic LLM Judges',
        'scope': 'Saved-score reproducibility; manuscript submitted separately',
        'files': [
            {'path': name, 'bytes': len(data),
             'sha256': hashlib.sha256(data).hexdigest()}
            for name, data in sorted(payload.items())
        ],
    }
    payload['MANIFEST.json'] = (json.dumps(manifest, indent=2) + '\n').encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output, 'w', compression=ZIP_DEFLATED, compresslevel=9) as archive:
        for name, data in sorted(payload.items()):
            archive.writestr(name, data)
    if output.stat().st_size > 25_000_000:
        raise RuntimeError('Reproducibility archive exceeds 25 MB')
    print(f'Wrote {output.name}: {output.stat().st_size:,} bytes, {len(payload)} files')


if __name__ == '__main__':
    main()
