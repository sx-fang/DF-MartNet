"""Static/source verification only; does not import numerical executors."""
from pathlib import Path
import ast
import configparser
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from dfm_repro.common import ROOT, json_read, sha256, package_digest, expression

def main():
    digest=package_digest();cases=json_read(ROOT/'configs/cases.json');count=0
    for p in ROOT.rglob('*.py'):
        if 'results' not in p.parts:ast.parse(p.read_text(encoding='utf-8-sig'),filename=str(p));count+=1
    for key,case in cases.items():
        for a in case['arms']:
            if sha256(ROOT/a['config'])!=a['config_sha256']:raise ValueError('Changed approved config '+key)
            src=ROOT/a['source'];meta=json_read(src/'SOURCE.json')
            for f in meta['files']:
                if sha256(src/f['file'])!=f['sha256']:raise ValueError('Changed source '+a['source'])
            if case['kind']=='official':
                argv=a['cli'];assert int(argv[argv.index('--seed')+1])==a['seed'];assert int(argv[argv.index('--max-iter')+1])==a['max_iter']
            else:
                c=configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation());c.read(ROOT/a['config'])
                assert c.getint('Environment','seed')==a['seed'];assert c.getint('Training','max_iter')==a['max_iter']
                for k in ['lr0_v','lr0_u','lr0_rho','decay_rate']:assert expression(c.get('Optimizer',k))>0
    corrected=json_read(ROOT/'corrected/SOURCE.json')
    for f in corrected['files']:
        assert sha256(ROOT/'corrected'/f['file'])==f['sha256']
    for record in json_read(ROOT/'data/ASSET_PROVENANCE.json'):
        assert (ROOT/record['release_file']).is_file()
        if record.get('release_sha256'):assert sha256(ROOT/record['release_file'])==record['release_sha256']
    print(f'PASS: {count} Python sources parsed; {len(cases)} case configurations; '+str(sum(len(c['arms']) for c in cases.values()))+' selected arms; package '+digest)

if __name__=='__main__':main()
