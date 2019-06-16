class Solution(object):
    def intersect(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: List[int]
        """
        if len(nums1)>len(nums2):
            return self.intersect(nums2,nums1)
        maps={}
        res=[]
        for num in nums1:
            if num not in maps:
                maps[num]=1
            else:
                maps[num]+=1
        for num in nums2:
            if num in maps:
                if maps[num]>0:
                    res.append(num)
                    maps[num]-=1
        return res
