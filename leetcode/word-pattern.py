class Solution(object):
    def wordPattern(self, pattern, str):
        """
        :type pattern: str
        :type str: str
        :rtype: bool
        """
        word = str.split()
        if len(word) != len(pattern):
            return False
        map_pattern,map_word = {},{}
        for i in xrange(len(word)):
            if pattern[i] not in map_pattern:
                map_pattern[pattern[i]] = word[i]
            if word[i] not in map_word:
                map_word[word[i]] = pattern[i]
            if map_word[word[i]] != pattern[i] or map_pattern[pattern[i]] != word[i]:
                return False
        return True
