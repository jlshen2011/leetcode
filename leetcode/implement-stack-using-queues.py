import collections
class Stack(object):
    def __init__(self):
        """
        initialize your data structure here.
        """
        self.data = collections.deque()

    def push(self, x):
        """
        :type x: int
        :rtype: nothing
        """
        self.data.append(x)
        for i in xrange(len(self.data) - 1):
            self.data.append(self.data.popleft())

    def pop(self):
        """
        :rtype: nothing
        """
        self.data.popleft()

    def top(self):
        """
        :rtype: int
        """
        return self.data[0]

    def empty(self):
        """
        :rtype: bool
        """
        return len(self.data) == 0
