class MinStack(object):

    def __init__(self):
        """
        initialize your data structure here.
        """
        self.stack = []
        self.min = None

    def push(self, x):
        """
        :type x: int
        :rtype: void
        """
        if not self.stack:
            self.min = x
            self.stack.append(0)
        else:
            self.stack.append(x - self.min)
            if x < self.min:
                self.min = x

    def pop(self):
        """
        :rtype: void
        """
        x = self.stack.pop()
        if x < 0:
            res = self.min
            self.min -= x
        else:
            res = x + self.min
        return res
        
    def top(self):
        """
        :rtype: int
        """
        x = self.stack[-1]
        if x < 0:
            return self.min
        else:
            return x + self.min

    def getMin(self):
        """
        :rtype: int
        """
        return self.min
