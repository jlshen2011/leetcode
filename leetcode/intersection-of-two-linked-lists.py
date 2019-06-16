class Solution(object):
    def getIntersectionNode(self, headA, headB):
        if not headA or not headB:
            return None
        pA, pB = headA, headB
        tailA, tailB = None, None
        while True:
            if not pA:
                pA = headB
            if not pB:
                pB = headA
            if not pA.next:
                tailA = pA
            if not pB.next:
                tailB = pB
            if tailA and tailB and tailA != tailB:
                return None
            if pA == pB:
                return pA
            pA, pB = pA.next, pB.next
