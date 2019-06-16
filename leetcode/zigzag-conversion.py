class Solution(object):
    def convert(self, s, numRows):
        """
        :type s: str
        :type numRows: int
        :rtype: str
        """
        if numRows == 1 or numRows >= len(s):
            return s
        zigzag = ["" for i in xrange(numRows)]
        row, step = 0, 1
        for x in s:
            zigzag[row] += x
            if row == 0:
                step = 1
            if row == numRows - 1:
                step = -1
            row += step
        res = ""
        for i in xrange(numRows):
            res += zigzag[i]
        return res
