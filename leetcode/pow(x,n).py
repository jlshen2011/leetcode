class Solution(object):
    def myPow(self, x, n):
        """
        :type x: float
        :type n: int
        :rtype: float
        """
        if n == 0:
            return 1
        if n < 0:
            return pow(1.0 / x, -n)
        if n % 2 == 1:
            return pow(x, n - 1) * x
        else:
            return pow(x * x, n / 2)
