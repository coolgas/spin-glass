import unittest
from adaptive_memory import GIB, growth_bytes, permitted


class MemoryTests(unittest.TestCase):
    def test_transient_peak_not_replaced_by_idle_rss(self):
        estimate=growth_bytes(GIB//5,32,64,peak_rss=2*GIB)
        self.assertEqual(estimate,12*GIB)
        self.assertFalse(permitted(GIB//5,estimate,100*GIB,8*GIB))

    def test_headroom_counts_only_additional_reservation(self):
        self.assertTrue(permitted(2*GIB,6*GIB,20*GIB,8*GIB))
        self.assertFalse(permitted(2*GIB,6*GIB,19*GIB,8*GIB))


if __name__=='__main__': unittest.main()
