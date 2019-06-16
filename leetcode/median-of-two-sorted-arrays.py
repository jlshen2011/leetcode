class Solution:
    def findMedianSortedArrays(self, A, B):
        def findKthItem(A, B, k):
            if len(A) > len(B):             
                A, B = B, A
            stepsA = (min(len(A), k) -1)/ 2
            # stepsB =  k - (stepsA + 1) -1 for the 0-based index
            stepsB = k - stepsA - 2
            if len(A) == 0:                 
                return B[k-1]
            elif k == 1:                    
                return min(A[0], B[0])
            elif A[stepsA] == B[stepsB]:    
                return A[stepsA]
            elif A[stepsA] > B[stepsB]:     
                return findKthItem(A, B[stepsB+1:], k-stepsB-1)
            else:                           
                return findKthItem(A[stepsA+1:], B, k-stepsA-1)
        if (len(A)+len(B))%2==1:
            return findKthItem(A, B, (len(A)+len(B))/2+1) * 1.0
        else:
            return (findKthItem(A, B, (len(A)+len(B))/2+1) + findKthItem(A, B, (len(A)+len(B))/2) ) / 2.0
