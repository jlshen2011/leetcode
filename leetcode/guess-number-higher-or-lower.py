class Solution(object):
    def guessNumber(self, n):
        """
        :type n: int
        :rtype: int
        """
        left, right = 1, n
        while True:
            mid = (left + right) / 2
            trial = guess(mid)
            if trial == -1:
                right = mid - 1
            elif trial == 1:
                left = mid + 1
            else:
                return mid
