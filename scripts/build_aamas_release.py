"""Build a portable saved-score archive from explicitly selected public files."""
from pathlib import Path
import hashlib
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'releases/aamas2027_reproducibility.zip'
FILES = [
    'README.md', 'LICENSE', 'requirements.txt',
    'docs/REPRODUCE.md', 'docs/RELEASE_AUDIT.md', 'docs/AI_ASSISTANCE.md',
    'docs/FILES_RATIONALE.md', 'docs/PROJECT_EXPLAINER.md',
    'paper/main.tex', 'paper/main.pdf', 'paper/references.bib',
    'paper/aamas.cls', 'paper/ACM-Reference-Format.bst', 'paper/by.pdf',
    'paper/figures/fig_overview.tex', 'paper/figures/fig1_gate_by_language.pdf',
    'paper/figures/fig2_interaction.pdf', 'paper/figures/fig3_gate_sweep.pdf',
    'scripts/build_aamas_release.py', 'scripts/generate_aamas_figures.py',
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
          if path.name != 'cbc_status.json']
FILES += [path.relative_to(ROOT).as_posix()
          for pattern in ('*.csv', '*.json', '*.jsonl')
          for path in (ROOT / 'paper/analysis/aamas_2027').rglob(pattern)
          if not path.name.endswith('_rerun.json')]

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
for name, expected in {
    'aamas.cls': 'e88c8e3e5fd1e39f93a2a4fa5b664e125b67487996b4236668206acf5c70ca7e',
    'ACM-Reference-Format.bst': '8ec002c927068bfc5b3cfe71b66aa4767b9e485530ac3c67ba5c064df4c2e6ac',
}.items():
    actual = hashlib.sha256((ROOT / 'paper' / name).read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError(f'{name} does not match the official AAMAS 2027 template')
overleaf_files = ['main.tex', 'references.bib', 'aamas.cls', 'ACM-Reference-Format.bst',
                 'by.pdf', 'figures/fig_overview.tex', 'figures/fig1_gate_by_language.pdf',
                 'figures/fig2_interaction.pdf', 'figures/fig3_gate_sweep.pdf']
with ZipFile(ROOT / 'releases/aamas2027_overleaf_upload.zip', 'w',
             compression=ZIP_DEFLATED, compresslevel=9) as archive:
    for name in overleaf_files:
        archive.write(ROOT / 'paper' / name, name)
    archive.writestr('README_OVERLEAF.txt',
        'Upload as a new Overleaf project. Select main.tex and pdfLaTeX.\n'
        'Keep the official aamas.cls and all figure files in their supplied locations.\n'
        'Use Recompile from scratch. OpenReview submission 1655.\n')
with ZipFile(OUTPUT, 'w', compression=ZIP_DEFLATED, compresslevel=9) as archive:
    for name in sorted(set(FILES)):
        path = ROOT / name
        if not path.is_file(): raise FileNotFoundError(path)
        data = path.read_bytes()
        # These records were sanitized when exported; guard against introducing
        # new private filesystem metadata into a future release.
        profile_roots = (b'C:' + b'\\Users\\', b'C:' + b'/Users/')
        if any(marker in data for marker in profile_roots):
            raise ValueError(f'Private filesystem path in {name}')
        archive.writestr(name, data)
if OUTPUT.stat().st_size > 25_000_000:
    raise RuntimeError('Reproducibility archive exceeds 25 MB')
print(f'Wrote {OUTPUT.name}: {OUTPUT.stat().st_size:,} bytes, {len(set(FILES))} files')
