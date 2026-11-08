import asyncio
import unittest
from memcut.native_memory.contract import MemoryObject, SearchResult
from memcut.source_anchored_v2.candidates import build_pool, candidate_protocol, verify_pool


class BudgetTests(unittest.TestCase):
    def test_native_search_receives_40_and_turn_cap_is_independent(self):
        class Bank:
            meta={'turn_order':[f't{i}' for i in range(100)]}
            calls=[]
            def state_identity(self):return 'frozen'
            def trace(self,oid):
                return self.roots[oid]
            async def search(self,q,*,top_k):
                self.calls.append(top_k)
                objs=[MemoryObject('fixture','fact',str(i),f'body{i}','v') for i in range(top_k)]
                self.roots={o.address:{f't{2*i}',f't{2*i+1}'} for i,o in enumerate(objs)}
                return SearchResult(q,{'retrieved_document':objs},costs={'native_seed_hits':[
                    {'address':o.address,'rank':i+1,'distance':float(i)} for i,o in enumerate(objs)]})
        bank=Bank()
        p=asyncio.run(build_pool(bank,'query',[],n_obj=40,n_turn=40));verify_pool(p)
        self.assertEqual(bank.calls,[40]);self.assertEqual(p['actual_objects'],40)
        self.assertEqual(p['candidate_turns_before_cap'],80);self.assertEqual(p['actual_candidate_turns'],40)
        q=asyncio.run(build_pool(bank,'query',[],n_obj=40,n_turn=20));verify_pool(q)
        self.assertEqual(q['actual_candidate_turns'],20)
        r=asyncio.run(build_pool(bank,'query',[]));self.assertEqual(bank.calls[-1],20)
        self.assertEqual(r['actual_candidate_turns'],20)

    def test_plan_freezes_both_candidate_units_without_changing_deletion_budget(self):
        from memcut.source_anchored_systems.runner import PROTOCOL
        forty=candidate_protocol(PROTOCOL,40,40)
        self.assertEqual((forty['n_obj'],forty['n_turn'],forty['k']),(40,40,10))
        self.assertEqual(forty['scoring'],PROTOCOL['scoring'])
        self.assertNotEqual(forty['version'],PROTOCOL['version'])
        for bad in [0,-1,True,20.5]:
            with self.assertRaises(ValueError):candidate_protocol(PROTOCOL,bad,40)


if __name__=='__main__':unittest.main()
