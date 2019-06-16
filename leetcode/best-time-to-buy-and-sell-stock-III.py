class Solution(object):
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        n = len(prices)
        if (n < 2):
            return 0
        st = prices[0]
        mp = []
        mprof = 0
        for i in range(0,n):
            if prices[i] - st > mprof:
                mprof = prices[i] - st
            if prices[i] < st:
                st = prices[i]
            mp.append(mprof)
        ed = prices[-1]
        mprof = 0
        ed = prices[-1]
        for i in range(n-1,-1,-1):
            if (ed - prices[i] + mp[i] > mprof):
                mprof = ed - prices[i] + mp[i]
            if (prices[i]>ed):
                ed = prices[i]
        return mprof
    
