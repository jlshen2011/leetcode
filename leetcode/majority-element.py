class Solution(object):
    def majorityElement(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        count = 0
        for i in xrange(len(nums)):
            if count == 0:
                tmp = nums[i]
                count = 1
            else:
                if nums[i] == tmp:
                    count += 1
                else:
                    count -= 1
        return tmp
