class Solution(object):
    def mergeTwoLists(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        dummy1, dummy2 = ListNode(0), ListNode(0)
        dummy1.next, dummy2.next = l1, l2
        p1, p2 = dummy1, dummy2
        while p1.next and p2.next:
            if p1.next.val <= p2.next.val:
                p1 = p1.next
            else:
                p1.next, p2.next.next, p2.next = p2.next, p1.next, p2.next.next
        if not p1.next:
            p1.next = p2.next
        return dummy1.next
