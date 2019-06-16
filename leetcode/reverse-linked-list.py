class Solution(object):
    def reverseList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        prev, curr, next = None, head, None
        while curr:
            next = curr.next
            curr.next = prev
            prev = curr
            curr = next
        return prev
