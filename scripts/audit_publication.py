"""Static inspection of the exact publication inputs; never prints matched values."""
from pathlib import Path
import argparse
import ast
import json
import os
import re
import sys
import zipfile
from urllib.parse import unquote

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from dfm_repro.common import json_read, package_digest

RULES={
    'private_key':re.compile(r'-----BEGIN (?:[A-Z0-9 ]+ )?PRIVATE KEY-----'),
    'access_token':re.compile(r'\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,}|AKIA[A-Z0-9]{16}|sk-[A-Za-z0-9_-]{20,})'),
    'credential_literal':re.compile(r'''(?i)\b(?:password|passwd|api_key|access_token|secret_key|client_secret|authorization)\b["']?\s*[:=]\s*["']([^"']+)["']'''),
    'private_path':re.compile(r'(?i)[A-Z]:[/\\](?:Users|OneDrive)[/\\]|/(?:scratch/users|home|users)/[A-Za-z0-9._-]+|\\\\[^\s\\]+\\'),
    'private_network':re.compile(r'(?<![\d.])(?:10\.(?:\d{1,3}\.){2}\d{1,3}|192\.168\.\d{1,3}\.\d{1,3}|172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3})(?![\d.])'),
    'internal_narrative':re.compile(r'(?i)\b'+('tun'+'ing')+r'\b|owner[- ]approved|pre[- ]committed|round[- ]\d|earlier_4run|console_\d{6}'),
}
PLACEHOLDER=re.compile(r'(?i)^(?:YOUR_[A-Z_]+|fake|example|placeholder|none|null|\*+|\{.*\}|\$\{.*\})$')
PRIVATE_FIELDS={'username','login','host','hostname','server','account','password','passwd','api_key','access_token','secret_key','client_secret','authorization'}
REMOVED_FIELDS={'historical_server','historical_jobid','regression_server','regression_jobid','affected_jobids','verified_representative','table_manifest_anchors_differ','excluded'}
SENSITIVE_NAME=re.compile(r'(?i)(?:^|/)(?:id_rsa|id_ed25519|authorized_keys|known_hosts|\.env)(?:$|\.)|\.(?:pem|key|p12|pfx)$')

def inspect_text(text, location, forbidden=(), public_contacts=(), public_attributions=()):
    findings=[]
    policies=[re.compile(r'(?<![A-Za-z0-9])'+re.escape(v)+r'(?![A-Za-z0-9])',re.IGNORECASE) for v in forbidden if v and not v.isdecimal()]
    definitions=set()
    if location=='scripts/audit_publication.py':
        for node in ast.parse(text).body:
            if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='RULES' for t in node.targets):
                definitions.update(range(node.lineno,node.end_lineno+1))
    for i,line in enumerate(text.splitlines(),1):
        policy_line=line
        for attribution in public_attributions:
            policy_line=policy_line.replace(attribution,'[public scholarly attribution]')
        for contact in public_contacts:
            if not re.fullmatch(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}',contact):continue
            policy_line=re.sub(r'(?<![A-Za-z0-9._%+@-])'+re.escape(contact)+r'(?![A-Za-z0-9._%+@-])','[public scholarly contact]',policy_line,flags=re.IGNORECASE)
        for name,pattern in RULES.items():
            if i in definitions and name in {'private_path','internal_narrative'}:continue
            match=pattern.search(line)
            if not match:continue
            if name=='credential_literal' and PLACEHOLDER.fullmatch(match.group(1)):continue
            findings.append(dict(file=location,line=i,rule=name))
        for pattern in policies:
            if i not in definitions and pattern.search(policy_line):
                findings.append(dict(file=location,line=i,rule='private_policy_literal'))
                break
    return findings

def inspect_json(value, location):
    findings=[]
    def visit(obj):
        if isinstance(obj,dict):
            for key,item in obj.items():
                k=key.casefold()
                if k in REMOVED_FIELDS:findings.append(dict(file=location,rule='private_history_field'))
                if k in PRIVATE_FIELDS and item is not None and str(item) and not PLACEHOLDER.fullmatch(str(item)):
                    findings.append(dict(file=location,rule='concrete_identity_or_credential_field'))
                visit(item)
        elif isinstance(obj,list):
            for item in obj:visit(item)
    visit(value)
    return findings

def inspect_npz(path, location, forbidden=()):
    """Inspect ZIP metadata and NPY headers/string arrays with standard library."""
    findings=[]
    with zipfile.ZipFile(path) as archive:
        findings+=inspect_text(archive.comment.decode('utf-8',errors='replace'),location,forbidden)
        for item in archive.infolist():
            name=item.filename
            if not name.endswith('.npy') or '/' in name or '\\' in name:
                findings.append(dict(file=location,rule='unexpected_binary_member'));continue
            data=archive.read(item)
            if data[:6]!=b'\x93NUMPY':
                findings.append(dict(file=location,rule='unexpected_binary_format'));continue
            version=data[6:8];end=10 if version==b'\x01\x00' else 12
            length=int.from_bytes(data[8:end],'little')
            header=data[end:end+length].decode('latin1' if version!=b'\x03\x00' else 'utf-8')
            descriptor=ast.literal_eval(header)['descr']
            if not isinstance(descriptor,str) or 'O' in descriptor:
                findings.append(dict(file=location,rule='opaque_binary_object'));continue
            findings+=inspect_text(name+'\n'+header,location,forbidden)
            if 'U' in descriptor:
                payload=data[end+length:].decode('utf-32-be' if descriptor.startswith('>') else 'utf-32-le')
                findings+=inspect_text(payload.replace('\0',' '),location,forbidden)
            elif 'S' in descriptor:
                findings+=inspect_text(data[end+length:].decode('utf-8',errors='replace').replace('\0',' '),location,forbidden)
    return findings

def inspect_pdf(path, location, forbidden=(), public_contacts=(), public_attributions=()):
    """Inspect text and all PDF object strings, including compressed objects."""
    try:
        from pypdf import PdfReader
        from pypdf.generic import IndirectObject
    except ImportError:
        return [dict(file=location,rule='missing_pdf_inspection_dependency')]
    findings=[]
    try:
        reader=PdfReader(path,strict=True)
        if reader.is_encrypted:return [dict(file=location,rule='encrypted_pdf')]
        if reader.attachments:findings.append(dict(file=location,rule='embedded_pdf_attachment'))
        for page in reader.pages:
            findings+=inspect_text(page.extract_text() or '',location,forbidden,public_contacts,public_attributions)
        def visit(value):
            if isinstance(value,dict):
                if value.get('/S') in {'/JavaScript','/Launch','/GoToR','/SubmitForm','/ImportData'} or '/JS' in value:
                    findings.append(dict(file=location,rule='active_or_external_pdf_action'))
                if value.get('/Type')=='/Filespec' or '/EF' in value:
                    findings.append(dict(file=location,rule='embedded_or_external_pdf_file'))
                if value.get('/Type')=='/Metadata' and hasattr(value,'get_data'):
                    visit(value.get_data().decode('utf-8',errors='replace'))
                for key,item in value.items():
                    visit(str(key))
                    if key!='/ID':visit(item)
            elif isinstance(value,(list,tuple)):
                for item in value:visit(item)
            elif isinstance(value,(str,bytes)):
                text=value.decode('utf-8',errors='replace') if isinstance(value,bytes) else str(value)
                findings.extend(inspect_text(unquote(text),location,forbidden,public_contacts,public_attributions))
        objects={(generation,number) for generation,entries in reader.xref.items() for number in entries if number}
        objects.update((0,number) for number in reader.xref_objStm)
        for generation,number in sorted(objects):visit(reader.get_object(IndirectObject(number,generation,reader)))
    except Exception:
        findings.append(dict(file=location,rule='pdf_parse_or_text_extraction_failed'))
    return list({json.dumps(item,sort_keys=True):item for item in findings}.values())


def publication_files(root):
    out=set()
    for directory,dirs,files in os.walk(root):
        dirs[:]=[d for d in dirs if d not in {'results','__pycache__','.venv','.git'}]
        for name in files:
            rel=(Path(directory)/name).relative_to(root).as_posix()
            if rel=='slurm/site.json' or name.endswith(('.pyc','.pyo')):continue
            out.add(rel)
    return out

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--private-policy',type=Path,help='Optional private JSON with forbidden_literals; keep outside the release')
    args=parser.parse_args()
    forbidden=json_read(args.private_policy).get('forbidden_literals',[]) if args.private_policy else []
    pins=json_read(ROOT/'PACKAGE_FILES.json');expected=set(pins)|{'PACKAGE_FILES.json'}
    digest=package_digest();findings=[]
    for rel in sorted(publication_files(ROOT)-expected):findings.append(dict(file=rel,rule='unlisted_publication_input'))
    snapshot=json_read(ROOT/'docs/paper_snapshot.json') if (ROOT/'docs/paper_snapshot.json').is_file() else {}
    public_contacts=snapshot.get('author_contact_emails',[])
    public_attributions=snapshot.get('public_attribution_text',[])
    binary=pdfs=0
    for rel in sorted(expected):
        path=ROOT/rel
        if SENSITIVE_NAME.search(rel):findings.append(dict(file=rel,rule='credential_filename'))
        if path.suffix=='.npz':
            binary+=1;findings+=inspect_npz(path,rel,forbidden);continue
        if path.suffix.lower()=='.pdf':
            pdfs+=1;findings+=inspect_pdf(path,rel,forbidden,public_contacts if rel=='docs/paper.pdf' else (),public_attributions if rel=='docs/paper.pdf' else ());continue
        try:text=path.read_text(encoding='utf-8-sig')
        except UnicodeError:
            findings.append(dict(file=rel,rule='unreviewed_binary'));continue
        findings+=inspect_text(text,rel,forbidden,public_contacts if rel=='docs/paper_snapshot.json' else (),public_attributions if rel=='docs/paper_snapshot.json' else ())
        if path.suffix=='.json':findings+=inspect_json(json.loads(text),rel)
    result=dict(package_sha256=digest,files=len(expected),npz_inspected=binary,pdf_inspected=pdfs,passed=not findings,findings=findings,
                scope='Locked publication files, PDF text/objects and binary-array headers; excludes private runtime outputs',
                limitation='Pattern inspection cannot prove absence of all possible secret formats')
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    display={k:v for k,v in result.items() if k!='findings'}
    display['finding_count']=len(findings);display['first_findings']=findings[:25]
    print(json.dumps(display,indent=2))
    return 0 if result['passed'] else 1

if __name__=='__main__':sys.exit(main())
