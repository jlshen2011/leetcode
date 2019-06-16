class Solution(object):
    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        maps = {}
        for i in xrange(len(nums)):
            if target - nums[i] not in maps:
                maps[nums[i]] = i
            else:
                return [i, maps[target - nums[i]]]
        return None
