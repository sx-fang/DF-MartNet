"""Metadata/control-flow tests with fake files; no training, inference or MC."""
from pathlib import Path
import sys
import tempfile
import unittest
from dfm_repro.common import summary, expression, fingerprints, validate_files, json_read, json_write
from dfm_repro.cli import completed_stage, resolve_cases, slurm_submit
from dfm_repro.collect import final_row, full_history

class Contracts(unittest.TestCase):
    def test_public_author_contact_does_not_whitelist_other_identity_text(self):
        from scripts.audit_publication import inspect_text
        contact='researcher'+'@example.org'
        self.assertFalse(inspect_text(contact,'fixture.pdf',['researcher'],[contact]))
        findings=inspect_text(contact+'\nlogin: researcher','fixture.pdf',['researcher'],[contact])
        self.assertEqual(findings,[dict(file='fixture.pdf',line=2,rule='private_policy_literal')])
        self.assertTrue(inspect_text(contact+'.private','fixture.pdf',['researcher'],[contact]))
        attribution='Research funded by Example Cluster.'
        self.assertFalse(inspect_text(attribution,'fixture.pdf',['cluster'],public_attributions=[attribution]))
        self.assertTrue(inspect_text(attribution+'\nlogin: cluster','fixture.pdf',['cluster'],public_attributions=[attribution]))

    def test_pdf_inspection_checks_object_strings_and_active_content(self):
        from types import SimpleNamespace
        import unittest.mock as mock
        from scripts.audit_publication import inspect_pdf
        secret='synthetic-local-account'
        objects={1:{'/Creator':secret},2:{'/S':'/JavaScript','/JS':'void(0)'},3:{'/Type':'/Filespec'}}
        reader=SimpleNamespace(is_encrypted=False,attachments={},pages=[SimpleNamespace(extract_text=lambda:'Public paper')],
            xref={0:{1:0,2:0}},xref_objStm={3:0},get_object=lambda ref:objects[ref[0]])
        modules={'pypdf':SimpleNamespace(PdfReader=lambda path,strict:reader),
            'pypdf.generic':SimpleNamespace(IndirectObject=lambda number,generation,reader:(number,generation))}
        with mock.patch.dict(sys.modules,modules):findings=inspect_pdf(Path('fixture.pdf'),'fixture.pdf',[secret])
        self.assertEqual({f['rule'] for f in findings},{'private_policy_literal','active_or_external_pdf_action','embedded_or_external_pdf_file'})
        self.assertNotIn(secret,str(findings))

    def test_pdf_inspection_requires_its_parser(self):
        import unittest.mock as mock
        from scripts.audit_publication import inspect_pdf
        with mock.patch.dict(sys.modules,{'pypdf':None}):
            self.assertEqual(inspect_pdf(Path('fixture.pdf'),'fixture.pdf'),[dict(file='fixture.pdf',rule='missing_pdf_inspection_dependency')])

    def test_publication_findings_do_not_echo_values(self):
        from scripts.audit_publication import inspect_text
        label='pass'+'word';value='synthetic-review-value'
        findings=inspect_text(label+" = '"+value+"'",'fixture.py')
        self.assertEqual(findings,[dict(file='fixture.py',line=1,rule='credential_literal')])
        self.assertNotIn(value,str(findings))
        self.assertFalse(inspect_text(label+" = 'YOUR_PASSWORD'",'fixture.py'))

    def test_publication_rejects_concrete_account_metadata(self):
        from scripts.audit_publication import inspect_json
        self.assertFalse(inspect_json({'account':'YOUR_ACCOUNT'},'site.example.json'))
        self.assertEqual(inspect_json({'account':'synthetic-local-account'},'fixture.json')[0]['rule'],'concrete_identity_or_credential_field')
        self.assertTrue(inspect_json({'historical_'+'jobid':42},'fixture.json'))

    def test_publication_policy_does_not_match_numeric_substrings(self):
        from scripts.audit_publication import inspect_text
        self.assertFalse(inspect_text('0.123456,1.23e01','fixture.csv',['e01','123']))
        self.assertTrue(inspect_text('"node": "e01"','fixture.json',['e01']))

    def test_publication_checks_binary_string_metadata_without_numpy(self):
        import zipfile
        from scripts.audit_publication import inspect_npz
        value='synthetic-local-account'
        header=repr({'descr':'<U'+str(len(value)),'fortran_order':False,'shape':(1,)})+'\n'
        encoded=header.encode('latin1')
        content=b'\x93NUMPY\x01\x00'+len(encoded).to_bytes(2,'little')+encoded+value.encode('utf-32-le')
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'fixture.npz'
            with zipfile.ZipFile(p,'w') as z:z.writestr('label.npy',content)
            findings=inspect_npz(p,'fixture.npz',[value])
            self.assertEqual(findings,[dict(file='fixture.npz',line=1,rule='private_policy_literal')])
            self.assertNotIn(value,str(findings))
        self.assertNotIn('numpy',sys.modules)
        self.assertNotIn('torch',sys.modules)

    def test_publication_inventory_excludes_private_runtime_outputs(self):
        from scripts.audit_publication import publication_files
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'slurm').mkdir();(root/'results').mkdir()
            (root/'slurm/site.json').write_text('private local settings')
            (root/'results/local.log').write_text('private local output')
            (root/'README.md').write_text('public documentation')
            (root/'unexpected.txt').write_text('must be reviewed')
            self.assertEqual(publication_files(root),{'README.md','unexpected.txt'})

    def test_final_is_not_best(self):
        rows=[{'it':'0','rel_l1err':'0.001'},{'it':'10','rel_l1err':'0.3'}]
        self.assertEqual(final_row(rows,10),rows[-1])
        with self.assertRaises(ValueError):final_row(rows,11)
        rows[-1]['rel_l1err']='nan'
        with self.assertRaises(ValueError):final_row(rows,10)

    def test_single_seed_has_no_cross_seed_sd(self):
        self.assertIsNone(summary([3])['sample_sd'])
        self.assertAlmostEqual(summary([1,3])['sample_sd'],2**.5)
        self.assertEqual(summary([1,3,20])['median'],3)
        with self.assertRaises(ValueError):summary([1,float('nan')])

    def test_fresh_history_requires_all_updates_and_native_final_rc(self):
        rows=[dict(it=i,rel_l1err=.1,rc=.2) for i in range(3)]
        full_history(rows,2,require_rc=True)
        with self.assertRaises(ValueError):full_history(rows[::2],2)
        rows[-1]['rc']=''
        with self.assertRaises(ValueError):full_history(rows,2,require_rc=True)

    def test_ini_expression_cannot_execute(self):
        self.assertAlmostEqual(expression('3 * 1e-3 / 10000**0.8'),1.892872033440579e-6)
        with self.assertRaises(ValueError):expression('__import__("os").getcwd()')

    def test_changed_artifact_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'result.txt';p.write_text('saved data');pins=fingerprints([p],d);validate_files(pins,d)
            p.write_text('other data')
            with self.assertRaises(ValueError):validate_files(pins,d)

    def test_resume_is_idempotent_and_input_pinned(self):
        with tempfile.TemporaryDirectory() as d:
            r=Path(d);p=r/'fake.txt';calls=[]
            def fake():calls.append(1);p.write_text('completed fake stage');return [p]
            completed_stage(r,'example',{'source':'A'},fake,False)
            completed_stage(r,'example',{'source':'A'},fake,True)
            self.assertEqual(len(calls),1)
            with self.assertRaises(ValueError):completed_stage(r,'example',{'source':'B'},fake,True)
            p.write_text('changed')
            with self.assertRaises(ValueError):completed_stage(r,'example',{'source':'A'},fake,True)

    def test_failed_stage_never_automatically_retries(self):
        with tempfile.TemporaryDirectory() as d:
            def fake():raise RuntimeError('fake failure')
            with self.assertRaises(RuntimeError):completed_stage(Path(d),'failed',{},fake,False)
            self.assertEqual(json_read(Path(d)/'stages/failed/stage.json')['state'],'failed')
            with self.assertRaises(RuntimeError):completed_stage(Path(d),'failed',{},fake,True)

    def test_resource_guard_precedes_numeric_imports(self):
        from dfm_repro.gpu_worker import gpu_check
        import unittest.mock as mock
        with mock.patch('dfm_repro.gpu_worker.sys.platform','win32'):
            with self.assertRaises(RuntimeError):gpu_check(8)
        self.assertNotIn('torch',sys.modules)

    def test_comparison_group_requires_both_methods(self):
        with self.assertRaises(ValueError):resolve_cases('comparison',{'compare_df_hjb2':{},'compare_df_hjb3':{}})

    def test_login_node_guard_precedes_torch(self):
        from dfm_repro.gpu_worker import gpu_check
        import unittest.mock as mock
        with mock.patch('dfm_repro.gpu_worker.sys.platform','linux'),mock.patch.dict('os.environ',{},clear=True):
            with self.assertRaises(RuntimeError):gpu_check(8)
        self.assertNotIn('torch',sys.modules)

    def test_slurm_resume_checks_terminal_and_final_log_bytes_without_resubmit(self):
        from types import SimpleNamespace
        import unittest.mock as mock
        with tempfile.TemporaryDirectory() as d:
            r=Path(d);(r/'raw').mkdir();data=r/'measure.csv';data.write_text('saved data')
            log=r/'raw/slurm_7654321.out';log.write_text('final scheduler output')
            site=r/'site.json';json_write(site,dict(partition='fake',account='fake',time='01:00:00',cpus_per_gpu=1,mem_gb=1))
            json_write(r/'slurm_job.json',dict(package='locked',jobid='7654321'))
            json_write(r/'run_manifest.json',dict(state='complete',artifacts=fingerprints([data],r)))
            with mock.patch('dfm_repro.cli.subprocess.check_output',side_effect=['','7654321|COMPLETED|0:0|00:42|fake|gres/gpu=8\n']) as queries:
                slurm_submit(r,[],{},SimpleNamespace(site=site,resume=True),'locked')
            self.assertEqual(queries.call_count,2)
            self.assertEqual([call.args[0][0] for call in queries.call_args_list],['squeue','sacct'])
            manifest=json_read(r/'run_manifest.json');self.assertTrue(manifest['slurm_terminal_verified'])
            validate_files(manifest['artifacts'],r)
            self.assertIn('raw/slurm_7654321.out',manifest['artifacts'])
            log.write_text('changed after acceptance')
            with self.assertRaises(ValueError):validate_files(manifest['artifacts'],r)

if __name__=='__main__':unittest.main()
