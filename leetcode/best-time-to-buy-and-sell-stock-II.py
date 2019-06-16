class Solution(object):
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        max_profit = 0
        for i in xrange(len(prices)-1):
            max_profit += max(0, prices[i+1] - prices[i])
        return max_profit
