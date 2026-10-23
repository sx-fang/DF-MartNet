"""Standard-library checks for the submitted experimental protocol."""
from .common import ROOT, json_read

HJB_CASES={'hjb1_d10000','hjb2_d10000','hjb3_d10000',
           'hjb3_d2000_w2010','hjb3_d2000_w10010'}

def evaluation_counts(case_key, case):
    """Global paper counts. Never multiply them by visible or allocated GPUs."""
    cost_paths=32768 if case_key in HJB_CASES else 4096
    if case_key.startswith('compare_') and case['kind']!='official':
        import configparser
        cfg=configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
        cfg.read(ROOT/case['arms'][0]['config'])
        cost_paths=cfg.getint('Example','M_rc',fallback=4096)
    return {'cost_paths':cost_paths,
            'deployment_paths':128 if case_key in HJB_CASES else 4096,
            'path_reference_paths':1000000 if case_key in HJB_CASES else None,
            'profile_points':101,'reference_seed':20260821,'path_seed':20260821}

def compare_own_policy(control, draw_states, reference, predict):
    """Pair predictions and references on states drawn by this very control."""
    states=draw_states(control)
    truth=reference(states)
    return states,truth,predict(states)

def require_complete_evaluation(arms, rows):
    expected=[a['id'] for a in arms]
    actual=[r['arm'] for r in rows]
    if len(actual)!=len(set(actual)) or set(actual)!=set(expected):
        raise ValueError('Evaluation must contain exactly the prescribed independent runs')

def validate_execution_sources(cases):
    """Check source manifests and the complete local import closure without torch."""
    import ast
    import sys
    from pathlib import Path
    from .common import sha256
    external={'numpy','pandas','scipy','matplotlib','psutil','torch','triton'}
    checked=set()
    for key,case in cases.items():
        for arm in case['arms']:
            rel=Path(arm['source'])
            if rel.is_absolute() or '..' in rel.parts:raise ValueError('Source outside package')
            source=ROOT/rel
            if not source.is_dir():raise FileNotFoundError(source)
            if case['kind']=='official':
                if not (source/'run.py').is_file():raise FileNotFoundError(source/'run.py')
                continue
            if str(rel) in checked:continue
            checked.add(str(rel))
            manifest=json_read(source/'SOURCE.json')
            for entry in manifest['files']:
                if sha256(source/entry['file'])!=entry['sha256']:raise ValueError('Changed execution module')
            local={p.stem for p in source.glob('*.py')}
            for file in source.glob('*.py'):
                for node in ast.walk(ast.parse(file.read_text(encoding='utf-8-sig'))):
                    modules=([node.module] if isinstance(node,ast.ImportFrom) and not node.level else
                             [n.name for n in node.names] if isinstance(node,ast.Import) else [])
                    for module in modules:
                        name=(module or '').split('.')[0]
                        if name and name not in local|external|sys.stdlib_module_names:
                            raise ValueError('Missing execution dependency '+name+' in '+str(file))
    return sorted(checked)
