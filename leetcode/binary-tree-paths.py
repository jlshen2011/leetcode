class Solution:
    # @param {TreeNode} root
    # @return {string[]}
    def binaryTreePaths(self, root):
        if not root:
            return []
        self.ans = []
        self.dfs(root, str(root.val))
        return self.ans
    
    def dfs(self, root, path):
        if not root.left and not root.right:
            self.ans.append(path)
        if root.left:
            self.dfs(root.left, path + "->" +str(root.left.val))
        if root.right:
            self.dfs(root.right, path + "->" +str(root.right.val))  
