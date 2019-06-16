class Solution(object):
    def countAndSay(self, n):
        """
        :type n: int
        :rtype: str
        """

        count = 1
        prev = "1"
        for j in xrange(1, n):
            res = ""
            count = 1
            for i in xrange(len(prev)):
                if i > 0:
                    if prev[i] == prev[i - 1]:
                        count += 1
                    else:
                        res += str(count)
                        res += str(prev[i - 1])
                        count = 1
                if i == len(prev) - 1:
                    res += str(count)
                    res += prev[i]
            prev = res
        return prev
