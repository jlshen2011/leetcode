class Solution(object):
    def getRow(self, rowIndex):
        """
        :type rowIndex: int
        :rtype: List[int]
        """
        result = [1] + [0] * rowIndex
        if rowIndex > 0:
            for i in xrange(1, rowIndex + 1):
                for j in reversed(xrange(1, i + 1)):
                    result[j] += result[j - 1]
        return result
