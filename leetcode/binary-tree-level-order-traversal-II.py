class Solution(object):
    def levelOrderBottom(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        if not root:
            return []
        res, current = [], [root]
        while current:
            val, next = [], []
            for node in current:
                val.append(node.val)
                if node.left:
                    next.append(node.left)
                if node.right:
                    next.append(node.right)
            res.append(val)
            current = next
        return res[::-1]
