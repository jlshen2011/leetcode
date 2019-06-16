class Solution(object):
    def reverseString(self, s):
        """
        :type s: str
        :rtype: str
        """
        res = []
        s = list(s)
        for i in xrange(len(s)):
            res.append(s.pop())
        return ''.join(res)
