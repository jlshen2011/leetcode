class Solution:
    # @param {integer[]} nums
    # @return {string}
    def largestNumber(self, nums):
        nums = [str(x) for x in nums]
        nums.sort(cmp = lambda x, y: cmp(y + x, x + y))
        largest = ''.join(nums)
        largest = largest.lstrip('0')
        if largest == '':
            return '0'
        else:
            return largest
