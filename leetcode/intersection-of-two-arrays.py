class Solution(object):
    def intersection(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: List[int]
        """
        if len(nums1)>len(nums2):
            return self.intersection(nums2,nums1)
        lookup=set()
        res=[]
        for num in nums1:
            if num not in lookup:
                lookup.add(num)
        for num in nums2:
            if num in lookup:
                res.append(num)
                lookup.discard(num)
        return res
