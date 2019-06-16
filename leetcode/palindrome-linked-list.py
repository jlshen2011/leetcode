class Solution(object):
    def isPalindrome(self, head):
        """
        :type head: ListNode
        :rtype: bool
        """
        if not head:
            return True
        slow, fast = head, head
        while fast.next and fast.next.next:
            slow, fast = slow.next, fast.next.next
        prev, curr, next = None, slow.next, None
        while curr:
            next = curr.next
            curr.next = prev
            prev = curr
            curr = next
        p1, p2 = prev, head
        while p1 and p1.val == p2.val:
            p1, p2 = p1.next, p2.next
        return not p1
