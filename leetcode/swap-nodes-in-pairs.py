class Solution(object):
    def swapPairs(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        dummy = ListNode(0)
        dummy.next = head
        p = dummy
        while p.next and p.next.next:
            p1, p2, p3 = p.next, p.next.next, p.next.next.next
            p1.next = p3
            p2.next = p1
            p.next = p2
            p = p.next.next
        return dummy.next
