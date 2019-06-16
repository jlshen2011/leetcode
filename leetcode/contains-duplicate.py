class Solution(object):
    def containsDuplicate(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        maps = {}
        for num in nums:
            if num not in maps:
                maps[num] = 1
            else:
                maps[num] += 1
                if maps[num] == 2:
                    return True
        return False
