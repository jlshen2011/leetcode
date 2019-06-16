import collections
class Solution(object):
    def isAnagram(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: bool
        """
        cs=collections.Counter(s)
        ct=collections.Counter(t)
        return (not cs-ct) and (not ct-cs)
