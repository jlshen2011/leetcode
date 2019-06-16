class Solution(object):
    def fizzBuzz(self, n):
        """
        :type n: int
        :rtype: List[str]
        """
        res = []
        for i in xrange(1, n + 1):
            if i % 3 == 0 and i % 5 != 0:
                res.append("Fizz")
            elif i % 3 != 0 and i % 5 == 0:
                res.append("Buzz")
            elif i % 3 == 0 and i % 5 == 0:
                res.append("FizzBuzz")
            else:
                res.append(str(i))
        return res
