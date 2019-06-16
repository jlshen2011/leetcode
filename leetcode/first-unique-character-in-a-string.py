class Solution(object):
    def firstUniqChar(self, s):
        """
        :type s: str
        :rtype: int
        """
        count = [0] * 26
        for l in s:
            count[ord(l) - 97] += 1
        for i, l in enumerate(s):
            if count[ord(l) - 97] == 1:
                return i
        return -1
