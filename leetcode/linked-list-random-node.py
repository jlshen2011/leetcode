from random import randint
class Solution(object):
    def __init__(self, head):
        """
        @param head The linked list's head.
        Note that the head is guaranteed to be not null, so it contains at least one node.
        :type head: ListNode
        """
        self.head = head
    def getRandom(self):
        """
        Returns a random node's value.
        :rtype: int
        """
        select = self.head.val
        current = self.head.next
        n = 1
        while current:
            select = current.val if randint(1, n + 1) == 1 else select
            current = current.next
            n += 1
        return select
