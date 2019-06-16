class Solution(object):
    def canJump(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        farthest=0
        for i, step in enumerate(nums):
            if i > farthest:
                break
            farthest = max(farthest, i + step)
        return farthest >= len(nums) - 1
