class Solution(object):
    def thirdMax(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        first, second, third = nums[0], float("-Inf"), float("-Inf")
        for i in xrange(1, len(nums)):
            if nums[i] > first:
                first, second,third = nums[i], first, second
            elif nums[i] < first and nums[i] > second:
                second, third = nums[i], second
            elif nums[i] < second and nums[i] > third:
                third = nums[i]
        return first if third == float("-Inf") else third
