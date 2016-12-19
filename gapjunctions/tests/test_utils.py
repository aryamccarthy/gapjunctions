import gapjunctions.utils as gutils

from nose.tools import assert_equal


class TestPairwise(object):
    """Unit test for the gapjunction.pairwise function.

    """

    def test_good_inputs(self):
        pairs = [x for x in gutils.pairwise(range(5))]
        assert_equal(pairs, [(0, 1), (1, 2), (2, 3), (3, 4)])
