class Solution(object):
    def romanToInt(self, s):
        """
        :type s: str
        :rtype: int
        """
        res = 0
        maps = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
        for i in xrange(len(s)):
            if i > 0 and maps[s[i]] > maps[s[i-1]]:
                res += maps[s[i]] - 2 * maps[s[i-1]]
            else:
                res += maps[s[i]]
        return res
