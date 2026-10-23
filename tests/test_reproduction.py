"""Metadata and existing-data regressions. No solver/network/MC execution."""
from pathlib import Path
import configparser
import importlib.util
import statistics
import sys
import tempfile
import unittest
from unittest.mock import patch

from dfm_repro.common import ROOT,json_read,read_csv
from dfm_repro.cli import freeze_sources
from dfm_repro.collect import path_archived,epsilon_profile_stats,archived_collect
from dfm_repro.protocol import (validate_execution_sources,compare_own_policy,
                                require_complete_evaluation,evaluation_counts)


class Reproduction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.cases=json_read(ROOT/'configs/cases.json')

    def test_frozen_source_has_every_selected_entry_and_dependency(self):
        validate_execution_sources(self.cases)
        with tempfile.TemporaryDirectory() as d:
            frozen=freeze_sources(Path(d),list(self.cases),self.cases)
            for case in self.cases.values():
                for arm in case['arms']:
                    entry='run.py' if case['kind']=='official' else 'solver.py'
                    self.assertTrue((frozen/arm['source']/entry).is_file())
            self.assertTrue((frozen/'data/archived/hjb_table/true_profiles.csv').is_file())

    def test_hjb_path_reader_matches_all_five_saved_independent_results(self):
        for key in ['hjb1_d10000','hjb2_d10000','hjb3_d10000','hjb3_d2000_w2010','hjb3_d2000_w10010']:
            with tempfile.TemporaryDirectory() as d:
                measured=path_archived(key,self.cases[key],Path(d))
                layers=read_csv(Path(d)/(key+'_path_layers.csv'))
            expected=read_csv(ROOT/'data/archived'/key/'published_path_perseed.csv')
            self.assertEqual(len(measured),5);self.assertEqual(len(layers),5*32)
            for actual,saved in zip(measured,expected):
                self.assertAlmostEqual(actual['re_path'],float(saved['RE_v']),places=13)

    def test_evaluation_never_reuses_another_controls_state_cloud(self):
        calls=[]
        def draw(control):return ('states-driven-by',control)
        def reference(states):calls.append(('reference',states));return ('truth-at',states)
        def predict(states):calls.append(('prediction',states));return ('value-at',states)
        one=compare_own_policy('policy one',draw,reference,predict)
        two=compare_own_policy('policy two',draw,reference,predict)
        self.assertNotEqual(one[0],two[0])
        self.assertEqual(calls[0][1],calls[1][1]);self.assertEqual(calls[2][1],calls[3][1])
        with self.assertRaises(ValueError):
            require_complete_evaluation([{'id':'one'},{'id':'two'}],[{'arm':'one'}])

    def test_epsilon_has_exactly_the_original_five_seeds_and_weights(self):
        for key in ['eps1','eps0p5','eps0p25','eps0p125','eps0']:
            arms=self.cases[key]['arms'];self.assertEqual([a['seed'] for a in arms],[1,2,3,4,5])
            for arm in arms:
                cfg=configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
                cfg.read(ROOT/arm['config'])
                self.assertEqual(cfg.getint('Environment','seed'),arm['seed'])
                self.assertEqual(cfg.getfloat('Training','w_term_v'),.5 if key in ['eps1','eps0'] else 1.)
                self.assertEqual(cfg.getint('Training','max_iter'),6000)
                self.assertEqual(cfg.getint('Training','num_xpts_batch'),6400)
                self.assertEqual(cfg.getint('Training','num_pilot_paths'),10240)

    def test_epsilon_plot_sd_is_across_all_five_saved_runs(self):
        samples=read_csv(ROOT/'data/archived/epsilon_sweep/per_seed_profiles.csv')
        for key,tag in [('eps1','eps1'),('eps0p5','eps2'),('eps0p25','eps4'),('eps0p125','eps8'),('eps0','eps0')]:
            data=[r for r in samples if r['case']==key];arms=[a['id'] for a in self.cases[key]['arms']]
            result=epsilon_profile_stats(data,arms)
            saved=read_csv(ROOT/f'data/archived/epsilon_sweep/cost_{tag}.csv')
            for actual,expected in zip(result,saved):
                self.assertEqual(actual['n'],5)
                self.assertAlmostEqual(actual['v_pred'],float(expected['v_pred']),places=13)
                self.assertAlmostEqual(actual['jhat'],float(expected['jhat']),places=13)
                self.assertAlmostEqual(actual['jhat_sd'],float(expected['jhat_std']),places=13)
            with self.assertRaises(ValueError):epsilon_profile_stats(data[:-1],arms)

    def test_rho_saved_data_collection_uses_its_native_final_costs(self):
        keys=['rho_'+str(n) for n in [75,150,300,600,1200]]
        with tempfile.TemporaryDirectory() as d,patch('dfm_repro.plots.generate'):
            run=Path(d);archived_collect(run,keys,self.cases,'rho_sweep')
            result=read_csv(run/'tables/rho_sweep.csv')
        for actual,key in zip(result,keys):
            values=[abs(float(read_csv(ROOT/'data/archived'/key/(a['id']+'_history.csv'))[-1]['rc'])) for a in self.cases[key]['arms']]
            self.assertAlmostEqual(float(actual['rc_mean']),statistics.mean(values),places=13)
            self.assertAlmostEqual(float(actual['rc_sd']),statistics.stdev(values),places=13)

    def test_table1_exports_paper_mean_runtime_separately_from_median(self):
        keys=['hjb1_d10000','hjb2_d10000','hjb3_d10000','hjb3_d2000_w2010','hjb3_d2000_w10010']
        with tempfile.TemporaryDirectory() as d,patch('dfm_repro.plots.generate'):
            run=Path(d);archived_collect(run,keys,self.cases,'hjb_table')
            result=read_csv(run/'tables/table1.csv');tex=(run/'tables/table1_rows.tex').read_text()
        printed=json_read(ROOT/'data/archived/TABLE1_PRINTED.json')
        for row,key in zip(result,keys):
            self.assertEqual(round(float(row['rt_solve_mean'])),printed[key]['rt_solve'])
            self.assertNotEqual(row['rt_solve_mean'],row['rt_solve_median'])
            self.assertEqual(row['re_path_n'],'5')

    def test_archived_scalar_path_metric_does_not_substitute_the_qq_interior_pool(self):
        with tempfile.TemporaryDirectory() as d,patch('dfm_repro.plots.generate'):
            run=Path(d);archived_collect(run,['scalar_control'],self.cases,'scalar_control')
            result=read_csv(run/'tables/per_seed.csv')
        for row in result:
            self.assertEqual(row['re_path'],row['rel_l1err_ualoop'])
            self.assertEqual(row['re_u_path'],row['rel_l1err_u_ualoop'])
            self.assertIn('re_v_qq_interior',row)
            self.assertIn('re_u_qq_interior',row)

    def test_global_budget_bookkeeping_never_rounds_up_or_drops_remainders(self):
        source=ROOT/'engines/runtime_4d30ed0/sample_budgets.py'
        spec=importlib.util.spec_from_file_location('fixture_budget_bookkeeping',source)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        for total,world in [(1000000,8),(17,4),(12800,4),(10000,8)]:
            shares=module.split_global_budget(total,world)
            self.assertEqual(sum(shares),total);self.assertLessEqual(max(shares)-min(shares),1)
        with self.assertRaises(ValueError):module.split_global_budget(3,4,minimum=1)
        self.assertEqual(evaluation_counts('hjb1_d10000',self.cases['hjb1_d10000'])['cost_paths'],32768)
        self.assertNotIn('torch',sys.modules)


if __name__=='__main__':unittest.main()
