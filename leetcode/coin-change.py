class Solution(object):
    def coinChange(self, coins, amount):
        """
        :type coins: List[int]
        :type amount: int
        :rtype: int
        """
        dp = [0] + [-1] * amount
        for x in xrange(amount):
            if dp[x] >= 0:
                for c in coins:
                    if x + c <= amount and  (dp[x + c] < 0 or dp[x + c] > dp[x] + 1):
                        dp[x + c] = dp[x] + 1
        return dp[amount]
