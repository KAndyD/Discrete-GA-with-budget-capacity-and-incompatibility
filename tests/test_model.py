import json
import unittest
import numpy as np
from common import ROOT
from model import generate,evaluate,repair,mutate,run

class TestEquipment(unittest.TestCase):
    def setUp(self):
        self.cfg=json.loads((ROOT/'config.json').read_text()); self.data=generate(self.cfg['data_seed'])
    def test_constraints_and_repair(self):
        data=self.data; cfg=self.cfg; rng=np.random.default_rng(1)
        for x in rng.integers(0,2,(100,36)):
            fixed=repair(x,data,cfg); self.assertEqual(evaluate(fixed,data,cfg)[1][0],0)
            np.testing.assert_array_equal(fixed,repair(fixed,data,cfg))
        self.assertGreater(evaluate(np.ones(36),data,cfg)[1][0],0)
        a,b=data['conflicts'][0]; x=np.zeros(36); x[[a,b]]=1
        self.assertGreater(evaluate(x,data,cfg)[1][0],0)
    def test_swap_preserves_cardinality(self):
        rng=np.random.default_rng(8); x=rng.random((50,36))<.3
        np.testing.assert_array_equal(x.sum(1),mutate(x,rng,'swap',.1).sum(1))
    def test_runs_feasible_reproducible_equal_budget(self):
        self.cfg['generations']=2
        for mode in ['repair','penalty','random']:
            a=run(self.cfg,self.data,8,mode); b=run(self.cfg,self.data,8,mode)
            self.assertEqual(a[3],3*self.cfg['population']); self.assertEqual(a[0],b[0])
            self.assertEqual(evaluate(a[1],self.data,self.cfg)[1][0],0)
            self.assertTrue(np.all(np.diff(a[2])>=0))
if __name__=='__main__': unittest.main()
