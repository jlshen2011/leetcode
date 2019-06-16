import collections
class Solution(object):
    def canConstruct(self, ransomNote, magazine):
        """
        :type ransomNote: str
        :type magazine: str
        :rtype: bool
        """
        countR = collections.Counter(ransomNote)
        countM = collections.Counter(magazine)
        return not countR - countM
