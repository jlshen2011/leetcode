class Solution(object):
    def containsNearbyDuplicate(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: bool
        """
        maps = {}
        for i, num in enumerate(nums):
            if num not in maps:
                maps[num] = i
            else:
                if i - maps[num] <= k:
                    return True
                maps[num] = i
        return False  
