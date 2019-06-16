class Queue(object):
    def __init__(self):
        """
        initialize your data structure here.
        """
        self.A, self.B = [], []
        
    def push(self, x):
        """
        :type x: int
        :rtype: nothing
        """
        self.A.append(x)

    def pop(self):
        """
        :rtype: nothing
        """
        self.peek()
        self.B.pop()

    def peek(self):
        """
        :rtype: int
        """
        if not self.B:
            while self.A:
                self.B.append(self.A.pop())
        return self.B[-1]

    def empty(self):
        """
        :rtype: bool
        """
        return not self.A and not self.B
