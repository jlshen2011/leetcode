class Solution(object):
    def numIslands(self, grid):
        """
        :type grid: List[List[str]]
        :rtype: int
        """
        if not grid:
            return 0
        row = len(grid)
        col = len(grid[0])
        visited = [[False for j in xrange(col)] for i in xrange(row)]
        count = 0
        for i in xrange(row):
            for j in xrange(col):
                if grid[i][j] == '1' and not visited[i][j]:
                    self.dfs(grid, visited, row, col, i, j)
                    count += 1
        return count
    
    def dfs(self, grid, visited, row, col, x, y):
        if grid[x][y] == '0' or visited[x][y]:
            return
        visited[x][y] = True
        if x != 0:
            self.dfs(grid, visited, row, col, x - 1, y)
        if x != row - 1:
            self.dfs(grid, visited, row, col, x + 1, y)
        if y != 0:
            self.dfs(grid, visited, row, col, x, y - 1)
        if y != col - 1:
            self.dfs(grid, visited, row, col, x, y + 1)
