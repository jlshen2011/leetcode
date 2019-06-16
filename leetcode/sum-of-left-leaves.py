class Solution(object):
    def sumOfLeftLeaves(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        res = 0
        if root:
            left, right = root.left, root.right
            if left and not left.left and not left.right:
                res += left.val
            res += self.sumOfLeftLeaves(left) + self.sumOfLeftLeaves(right)
        return res
