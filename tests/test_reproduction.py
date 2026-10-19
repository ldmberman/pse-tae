"""Meaningful guards for checkpoint evaluation and train/validation/test separation."""
import argparse,json,sys,tempfile,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.ensemble_predictions import combine
from scripts.evaluate_checkpoint import check_fold, model_arguments, parser as evaluation_parser
from scripts.frozen_pair_probes import partitions
from scripts.reproduce import command

class ReproductionTests(unittest.TestCase):
    def test_geometry_is_not_unpacked_as_pair_at_batch_size_two(self):
        import torch
        from models.stclassifier import PseTae
        torch.manual_seed(1)
        model=PseTae(positions=list(range(24))).eval()
        pixels=torch.randn(2,24,10,6);mask=torch.ones(2,24,6);geometry=torch.randn(2,24,4)
        with torch.no_grad():
            together=model(((pixels,mask),geometry))
            separate=torch.cat([model(((pixels[i:i+1],mask[i:i+1]),geometry[i:i+1])) for i in range(2)])
        torch.testing.assert_close(together,separate,atol=1e-5,rtol=1e-5)
    def test_original_partitions_disjoint_and_complete(self):
        for fold in range(1,6):
            s=partitions(191412,fold)
            self.assertEqual(sorted(np.concatenate(list(s.values())).tolist()),list(range(191412)))
            for a,b in [('train','val'),('train','test'),('val','test')]:self.assertEqual(len(np.intersect1d(s[a],s[b])),0)
    def test_checkpoint_fold_guards(self):
        check_fold({'protocol':{'fold':1}},'snapshot.pt',1)
        check_fold({},'checkpoints/tgt/run-a/fold-1/best.pt',1)
        for ck,path in [({'protocol':{'fold':2}},'snapshot.pt'),({},'Fold_2/model.pt'),({},'fold-2/best.pt'),({},'fold-unknown/best.pt'),({},'unknown.pt')]:
            with self.assertRaises(ValueError):check_fold(ck,path,1)
    def test_probability_average_and_reject_bad_alignment(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);y=np.arange(50)%20;pids=np.arange(50).astype(str);ids=partitions(50,1)['test']
            np.savez(d/'fold_1_features.npz',labels=y,pids=pids)
            a=np.full((len(ids),20),.05);b=a.copy();b[:,0]+=.1;b[:,1]-=.04;b/=b.sum(1,keepdims=True)
            paths=[]
            for n,q in [('a',a),('b',b)]:
                path=d/f'{n}.npz';np.savez(path,indices=ids,labels=y[ids],probabilities=q,fold=1,split='test');paths.append(path)
            result=combine(paths,1,'test',d);np.testing.assert_allclose(result[3],(a+b)/2)
            np.savez(paths[1],indices=ids[::-1],labels=y[ids],probabilities=b,fold=1,split='test')
            with self.assertRaises(ValueError):combine(paths,1,'test',d)
            np.savez(paths[1],indices=ids,labels=y[ids],probabilities=b,fold=2,split='test')
            with self.assertRaises(ValueError):combine(paths,1,'test',d)
    def test_native_fold_end_is_requested_fold(self):
        a=argparse.Namespace(experiment='baseline',action='train',fold=3,baseline=None,source=None,checkpoint=None,native='data/native',patch='data/patch',cache='results/pair_probes',device='cuda',workers=0,epochs=None,batch_size=None)
        cmd=command(a);self.assertEqual(cmd[cmd.index('--start_fold')+1],'3');self.assertEqual(cmd[cmd.index('--fold_count')+1],'3')
    def test_native_evaluation_uses_central_recipe(self):
        recipes=json.loads((ROOT/'configs/experiments.json').read_text())
        for name,recipe in recipes.items():
            if recipe['kind']!='native':continue
            with self.subTest(experiment=name):
                a=argparse.Namespace(experiment=name,action='evaluate',fold=1,baseline=None,source=None,checkpoint=None,native='data/native',patch='data/patch',cache='results/pair_probes',device='cpu',workers=0,batch_size=2,split='test',limit=0)
                args=evaluation_parser().parse_args(command(a)[2:])
                self.assertEqual(model_arguments(args),recipe['model_kwargs'])
                self.assertEqual(args.legacy_unscaled_kurtosis,name=='moments')
                args.family='patch'
                with self.assertRaises(ValueError):model_arguments(args)
    def test_custom_evaluation_config_still_supported(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'custom.json';path.write_text('{"parcel_stats":"mean_std"}')
            a=argparse.Namespace(experiment=None,model_config=str(path))
            self.assertEqual(model_arguments(a),{'parcel_stats':'mean_std'})
    def test_all_recipe_cli_flags(self):
        # Parse the actual trainer arguments; stop before executing any training.
        import runpy
        recipes=json.loads((ROOT/'configs/experiments.json').read_text())
        original=argparse.ArgumentParser.parse_args;old_argv=sys.argv
        class Parsed(Exception):pass
        def parse_then_stop(parser,*args,**kwargs):
            original(parser,*args,**kwargs)
            raise Parsed()
        try:
            argparse.ArgumentParser.parse_args=parse_then_stop
            for experiment in recipes:
                a=argparse.Namespace(experiment=experiment,action='train',fold=1,baseline=None,source=None,checkpoint=None,native='data/native',patch='data/patch',cache='results/pair_probes',device='cpu',workers=0,epochs=1,batch_size=2,teacher='teacher.npz')
                cmd=command(a);sys.argv=cmd[1:]
                with self.subTest(experiment=experiment):
                    with self.assertRaises(Parsed):runpy.run_path(str(ROOT/cmd[1]),run_name='__main__')
        finally:argparse.ArgumentParser.parse_args=original;sys.argv=old_argv
if __name__=='__main__':unittest.main()
