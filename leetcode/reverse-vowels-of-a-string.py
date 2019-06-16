class Solution(object):
    def reverseVowels(self, s):
        """
        :type s: str
        :rtype: str
        """
        l = list(s)
        i, j = 0, len(l) - 1
        while i < j:
            while i < j and l[i] not in 'aeiouAEIOU':
                i += 1
            while i < j and l[j] not in 'aeiouAEIOU':
                j -= 1
            if i < j:
                l[i], l[j] = l[j], l[i]
                i += 1
                j -= 1
        return "".join(l)
