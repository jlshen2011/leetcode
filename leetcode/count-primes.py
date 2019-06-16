class Solution(object):
    def countPrimes(self, n):
        """
        :type n: int
        :rtype: int
        """
        num = [1] * n
        i = 2
        while i * i < n:
            j = 2
            while j * i < n:
                num[j * i] = 0
                j += 1
            i += 1
            while num[i] == 0 and i * i < n:
                i += 1
        return sum(num[2:])
