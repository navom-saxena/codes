import unittest

import Case1


class test_sum(unittest.TestCase):
    def testUserRating(self):
        """
        Test that it can sum a list of integers
        """
        ob = Case1.Movies()
        data = ["1,6,2,980730861", "1,6,2,980730861", "1,6,2,980730861", "2,6,2,980730861",
                "2,6,2,980730861", "2,6,2,980730861"];
        result = ob.userRating(data)
        expected_count = 4
        for (k, v) in result.collect():
            getcount = v
            self.assertEqual(getcount, expected_count)


if __name__ == '__main__':
    unittest.main()
