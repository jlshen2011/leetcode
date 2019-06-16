class Solution(object):
    def isBalanced(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        return self.getHeight(root) >= 0
    def getHeight(self, root):
        if not root:
            return 0
        left_height, right_height = self.getHeight(root.left), self.getHeight(root.right)
        if left_height < 0 or right_height< 0 or abs(left_height - right_height) > 1:
            return -1
        return max(left_height, right_height) + 1
