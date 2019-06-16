class Solution(object):
    def isValid(self, s):
        """
        :type s: str
        :rtype: bool
        """
        stack, maps = [], {"(": ")", "[": "]", "{": "}"}
        for p in s:
            if p in maps:
                stack.append(p)
            elif len(stack) == 0 or maps[stack.pop()] != p:
                return False
        return len(stack) == 0
