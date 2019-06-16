class Solution(object):
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        local_smallest, max_profit = float("inf"), 0
        for i in xrange(len(prices)):
            if max_profit < prices[i] - local_smallest:
                max_profit = prices[i] - local_smallest
            if prices[i] < local_smallest:
                local_smallest = prices[i]
        return max_profit  
