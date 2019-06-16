class Solution(object):
    def isSymmetric(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        if not root:
            return True
        return self.isSymmetricRecu(root.left, root.right)
        
    def isSymmetricRecu(self, left, right):
        if not left and not right:
            return True
        if (not left and right) or (left and not right) or left.val != right.val:
            return False
        return self.isSymmetricRecu(left.left, right.right) and self.isSymmetricRecu(left.right, right.left)
