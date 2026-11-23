import unittest
from adaptive_d_policy import growth_reason, final_window_pass, locally_stable


def rows(count, D=128, weight=0., step=0.):
    return [dict(D=D, mixer=False, max_bond=D, discarded_weight=weight,
                 energy=-100+i*step) for i in range(count)]


class PolicyTests(unittest.TestCase):
    def test_stationary_energy_does_not_hide_truncation(self):
        self.assertEqual(growth_reason(rows(20, weight=2e-6), {}, 128, 20, 108),
                         ['persistent_truncation'])

    def test_fast_progress_does_not_trigger_growth(self):
        h=rows(40)
        for i in range(30, 40):
            h[i]['energy']=h[i-1]['energy']+(1e-4 if i<35 else 1e-5)
        self.assertEqual(growth_reason(h, {}, 128, 40, 108), [])

    def test_stall_at_cap_and_mixer_exclusion(self):
        h=rows(40, step=2e-6)
        self.assertIn('energy_stall',growth_reason(h, {}, 128, 40, 108))
        h[-1]['mixer']=True
        self.assertEqual(growth_reason(h, {}, 128, 40, 108), [])

    def test_no_growth_after_sixty(self):
        self.assertEqual(growth_reason(rows(80, weight=1e-3), {}, 128, 80, 108), [])

    def test_confirmation_cannot_cross_dimension_or_mixer(self):
        h=rows(120)
        m={100:dict(D=128,mixer=False,m=[0.]),120:dict(D=128,mixer=False,m=[0.])}
        self.assertTrue(final_window_pass(h,m,120,108))
        self.assertTrue(final_window_pass(h,m,120,108))
        h[-51]['D']=64
        self.assertFalse(final_window_pass(h,m,120,108))
        h[-51]['D']=128; h[-51]['mixer']=True
        self.assertFalse(final_window_pass(h,m,120,108))

    def test_unsaturated_bonds_do_not_trigger_growth(self):
        h=rows(20, weight=1e-3)
        for r in h: r['max_bond']=64
        self.assertEqual(growth_reason(h,{},128,20,108),[])

    def test_disagreement_requires_stable_magnetization(self):
        m={35:dict(D=128,mixer=False,m=[0.]),40:dict(D=128,mixer=False,m=[.1])}
        self.assertFalse(locally_stable(rows(40),m,40,108))
        m[40]['m']=[0.]
        self.assertTrue(locally_stable(rows(40),m,40,108))

    def test_nan_cannot_pass_confirmation(self):
        h=rows(120); h[-20]['energy']=float('nan')
        m={100:dict(D=128,mixer=False,m=[0.]),120:dict(D=128,mixer=False,m=[0.])}
        self.assertFalse(final_window_pass(h,m,120,108))

    def test_increase_invalidates_stall_window(self):
        h=rows(40,weight=0.,step=2e-6)
        h[-10]['D']=64
        self.assertEqual(growth_reason(h,{},128,40,108),[])


if __name__ == '__main__': unittest.main()
