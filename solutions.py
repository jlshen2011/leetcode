import heapq
import math
import itertools
from collections import *
from functools import cmp_to_key
from string import ascii_lowercase


class Solution:
    # 1. Two Sum
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        dic = {}
        for i in range(len(nums)):
            if nums[i] in dic:
                return [dic[nums[i]], i]
            dic[target - nums[i]] = i
        return [-1, -1]


    # 2. Add Two Numbers
    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        dummy = ListNode()
        node = dummy
        carry = 0
        while l1 and l2:
            tmp = l1.val + l2.val + carry
            carry = tmp // 10
            tmp = tmp % 10
            node.next = ListNode(tmp)
            l1 = l1.next
            l2 = l2.next
            node = node.next
        while l1:
            tmp = l1.val + carry
            carry = tmp // 10
            tmp = tmp % 10
            node.next = ListNode(tmp)
            l1 = l1.next
            node = node.next
        while l2:
            tmp = l2.val + carry
            carry = tmp // 10
            tmp = tmp % 10
            node.next = ListNode(tmp)
            l2 = l2.next
            node = node.next
        if carry:
            node.next = ListNode(1)
        return dummy.next


    # 3. Longest Substring Without Repeating Characters
    def lengthOfLongestSubstring(self, s: str) -> int:
        last_idx = {}
        ans = curr_start = 0
        for i in range(len(s)):
            if s[i] in last_idx:
                curr_start = max(last_idx[s[i]] + 1, curr_start)
            last_idx[s[i]] = i
            ans = max(ans, i - curr_start + 1)
        return ans


    # 4. Median of Two Sorted Arrays
    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        def kthSmallest(nums1, nums2, k):  # including 0th
            if not nums1:
                return nums2[k]
            if not nums2:
                return nums1[k]
            mid1, mid2 = len(nums1) // 2, len(nums2) // 2
            val1, val2 = nums1[mid1], nums2[mid2]
            if k > mid1 + mid2:
                if val1 < val2:
                    return kthSmallest(nums1[mid1 + 1 :], nums2, k - mid1 - 1)
                else:
                    return kthSmallest(nums1, nums2[mid2 + 1 :], k - mid2 - 1)
            else:
                if val1 < val2:
                    return kthSmallest(nums1, nums2[:mid2], k)
                else:
                    return kthSmallest(nums1[:mid1], nums2, k)
        lens = len(nums1) + len(nums2)
        if lens % 2 == 1:
            return kthSmallest(nums1, nums2, lens // 2)
        else:
            return (
                kthSmallest(nums1, nums2, lens // 2 - 1)
                + kthSmallest(nums1, nums2, lens // 2)
            ) / 2


    # 5. Longest Palindromic Substring
    def longestPalindrome(self, s: str) -> str:
        def expand(i, j):
            l, r = i, j
            k = 0
            while l - k >= 0 and r + k < n and s[l - k] == s[r + k]:
                k += 1
            return k
        if not s:
            return s
        n = len(s)
        start = end = 0
        max_len = 1
        for i in range(n):
            len1 = expand(i, i)
            len2 = expand(i, i + 1)
            if max_len < 2 * len1 - 1:
                start, end = i - (len1 - 1), i + (len1 - 1)
                max_len = 2 * len1 - 1
            if max_len < 2 * len2:
                start, end = i - len2 + 1, i + len2
                max_len = 2 * len2
        return s[start : end + 1]


    # 6. ZigZag Conversion
    def convert(self, s: str, numRows: int) -> str:
        if numRows == 1 or numRows >= len(s):
            return s
        zigzag = ["" for i in xrange(numRows)]
        row, step = 0, 1
        for x in s:
            zigzag[row] += x
            if row == 0:
                step = 1
            if row == numRows - 1:
                step = -1
            row += step
        res = ""
        for i in xrange(numRows):
            res += zigzag[i]
        return res


    # 7. Reverse Integer
    def reverseInteger(self, x: int) -> int:
        rev = 0
        ind = 1
        if x < 0:
            ind = -1
            x = x * ind
        while x != 0:
            temp = x % 10
            if rev > float("inf") / 10 or rev < float("-inf") / 10:
                return 0
            rev = rev * 10 + temp
            x = x // 10
        if rev < -(2 ** 31):
            return 0
        if rev > 2 ** 31 - 1:
            return 0
        return rev * ind


    # 8. String to Integer (atoi)
    def myAtoi(self, str: str) -> int:
        str = str.strip()
        if not str:
            return 0
        sign = -1 if str[0] == "-" else 1
        n = len(str)
        res = i = 0
        if str[0] in "+-":
            i = 1
        while i < n and str[i].isdigit():
            res = 10 * res + int(str[i])
            i += 1
        return max(-(2 ** 31), min(sign * res, 2 ** 31 - 1))


    # 9. Palindrome Number    
    def isPalindrome(self, x: int) -> bool:
        if x < 0:
            return False
        copy, reverse = x, 0
        while copy:
            reverse *= 10
            reverse += copy % 10
            copy /= 10
        return x == reverse


    # 10. Regular Expression Matching
    def isMatch(self, s: str, p: str) -> bool:
        if len(p) == 0:
            return len(s) == 0
        if len(p) == 1:
            return len(s) == 1 and (s[0] == p[0] or p[0] == ".")
        if p[1] != "*":
            return (
                len(s) > 0
                and (s[0] == p[0] or p[0] == ".")
                and self.isMatch(s[1:], p[1:])
            )
        while len(s) > 0 and (s[0] == p[0] or p[0] == "."):
            if self.isMatch(s, p[2:]):
                return True
            s = s[1:]
        return self.isMatch(s, p[2:])


    # 11. Container With Most Water
    def maxArea(self, height: List[int]) -> int:
        left, right = 0, len(height) - 1
        ans = 0
        while left < right:
            ans = max(ans, min(height[left], height[right]) * (right - left))
            if height[left] < height[right]:
                left += 1
            else:
                right -= 1
        return ans


    # 12. Integer to Roman
    def intToRoman(self, num: int) -> str:
        digits = [
            (1000, "M"),
            (900, "CM"),
            (500, "D"),
            (400, "CD"),
            (100, "C"),
            (90, "XC"),
            (50, "L"),
            (40, "XL"),
            (10, "X"),
            (9, "IX"),
            (5, "V"),
            (4, "IV"),
            (1, "I"),
        ]
        roman_digits = []
        # Loop through each symbol.
        for value, symbol in digits:
            # We don't want to continue looping if we're done.
            if num == 0:
                break
            count, num = divmod(num, value)
            # Append "count" copies of "symbol" to roman_digits.
            roman_digits.append(symbol * count)
        return "".join(roman_digits)


    # 13. Roman to Integer
    def romanToInt(self, s: str) -> int:
        roman = {
            "I": 1,
            "V": 5,
            "X": 10,
            "L": 50,
            "C": 100,
            "D": 500,
            "M": 1000,
            "IV": 4,
            "IX": 9,
            "XL": 40,
            "XC": 90,
            "CD": 400,
            "CM": 900,
        }
        i = total = 0
        n = len(s)
        while i < n:
            if i + 1 < n and s[i : i + 2] in roman:
                total += roman[s[i : i + 2]]
                i += 2
            else:
                total += roman[s[i]]
                i += 1
        return total
    

    # 14. Longest Common Prefix
    def longestCommonPrefix(self, strs: List[str]) -> str:
        if not strs:
            return ""
        res = ""
        for i in range(len(strs[0])):
            char = strs[0][i]
            for j in range(1, len(strs)):
                if len(strs[j]) < i + 1 or strs[j][i] != char:
                    return res
            res += char
        return res


    # 15. 3Sum
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        if not nums:
            return []
        ans = []
        nums.sort()
        n = len(nums)
        for i in range(n - 2):
            if i > 0 and nums[i] == nums[i - 1]:
                continue
            j, k = i + 1, n - 1
            while j < k:
                sum_ = nums[i] + nums[j] + nums[k]
                if sum_ < 0:
                    j += 1
                elif sum_ > 0:
                    k -= 1
                else:
                    ans.append([nums[i], nums[j], nums[k]])
                    while j < k and nums[j] == nums[j + 1]:
                        j += 1
                    while j < k and nums[k] == nums[k - 1]:
                        k -= 1
                    j += 1
                    k -= 1
        return ans


    # 16. 3Sum Closest
    def threeSumClosest(self, nums: List[int], target: int) -> int:
        nums.sort()
        n = len(nums)
        abs_diff = float("inf")
        for i in range(n - 2):
            j, k = i + 1, n - 1
            while j < k:
                sum_ = nums[i] + nums[j] + nums[k]
                diff = sum_ - target
                if diff < 0:
                    j += 1
                elif diff > 0:
                    k -= 1
                else:
                    return target
                if abs(diff) < abs_diff:
                    abs_diff = abs(sum_ - target)
                    ans = sum_
        return ans


    #17. Letter Combinations of a Phone Number
    def letterCombinations(self, digits: str) -> List[str]:
        dic = {
            "2": "abc",
            "3": "def",
            "4": "ghi",
            "5": "jkl",
            "6": "mno",
            "7": "pqrs",
            "8": "tuv",
            "9": "wxyz",
        }
        if not digits:
            return []
        n = len(digits)
        ans = []
        def helper(idx, path):
            if idx == n:
                ans.append(path)
                return
            for d in dic[digits[idx]]:
                helper(idx + 1, path + d)
        helper(0, "")
        return ans


    # 19. Remove Nth Node From End of List
    def removeNthFromEnd(self, head: ListNode, n: int) -> ListNode:
        if not head:
            return head
        p1 = ListNode(0, head)
        for _ in range(n):
            p1 = p1.next
        p2 = ListNode(0, head)
        dummy = p2
        while p1.next:
            p1 = p1.next
            p2 = p2.next
        p2.next = p2.next.next
        return dummy.next


    # 20. Valid Parentheses
    def isValid(self, s: str) -> bool:
        stack = []
        paren_map = {")": "(", "]": "[", "}": "{"}
        for c in s:
            if c not in paren_map:
                stack.append(c)
            elif not stack or paren_map[c] != stack.pop():
                return False
        return not stack


    # 21. Merge Two Sorted Lists
    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
        prev = dummy = ListNode()
        while l1 and l2:
            if l1.val < l2.val:
                prev.next = l1
                l1 = l1.next
                prev = prev.next
            else:
                prev.next = l2
                l2 = l2.next
                prev = prev.next
        if l1:
            prev.next = l1
        if l2:
            prev.next = l2
        return dummy.next


    # 22. Generate Parentheses
    def generateParenthesis(self, n: int) -> List[str]:
        def helper(left, right, path):
            if left == right == n:
                ans.append(path)
                return
            if left <= n - 1:
                helper(left + 1, right, path + "(")
            if right < left:
                helper(left, right + 1, path + ")")
        ans = []
        helper(0, 0, "")
        return ans


    # 23. Merge k Sorted Lists
    def mergeKLists(self, lists: List[ListNode]) -> ListNode:
        h = []
        for i, list in enumerate(lists):
            if list:
                heapq.heappush(h, [list.val, i])
        l = dummy = ListNode()
        while h:
            val, i = heapq.heappop(h)
            l.next = lists[i]
            if lists[i].next:
                heapq.heappush(h, [lists[i].next.val, i])
                lists[i] = lists[i].next
            l = l.next
        return dummy.next


    # 24. Swap Nodes in Pairs
    def swapPairs(self, head: ListNode) -> ListNode:
        dummy = ListNode(next=head)
        prev, curr = dummy, head
        while curr and curr.next:
            next = curr.next
            second_next = next.next
            prev.next = next
            curr.next = second_next
            next.next = curr
            prev = curr
            curr = second_next
        return dummy.next


    # 25. Reverse Nodes in k-Group
    def reverseKGroup(self, head: ListNode, k: int) -> ListNode:
        jump = dummy = ListNode(next=head)
        left = right = head
        while True:
            count = 0
            while right and count < k:
                count += 1
                right = right.next
            if count < k:
                return dummy.next
            else:
                prev, cur = right, left
                for _ in range(k):
                    next = cur.next
                    cur.next = prev
                    prev = cur
                    cur = next
                jump.next = prev
                jump = left
                left = right


    # 26. Remove Duplicates from Sorted Array
    def removeDuplicates(self, nums: List[int]) -> int:
        i = j = 1
        while i < len(nums):
            if nums[i] != nums[j - 1]:
                nums[j] = nums[i]
                j += 1
            i += 1
        return j


    # 27. Remove Element
    def removeElement(self, nums: List[int], val: int) -> int:
        j = 0
        for i in range(len(nums)):
            if nums[i] != val:
                nums[j] = nums[i]
                j += 1
        return j


    # 28. Implement strStr()
    def strStr(self, haystack: str, needle: str) -> int:
        m = len(needle)
        for i in range(0, len(haystack) - m + 1):
            if haystack[i : i + m] == needle:
                return i
        return -1


    # 29. Divide Two Integers
    def divide(self, dividend: int, divisor: int) -> int:
        if dividend == -2147483648 and divisor == -1:
            return 2147483647
        sign = -1 if (dividend < 0) ^ (divisor < 0) else 1
        dividend, divisor = abs(dividend), abs(divisor)
        ans = 0
        if divisor == 1:
            ans = dividend
        else:
            while dividend >= divisor:
                x = divisor
                y = 1
                while dividend >= (x << 1):
                    x <<= 1
                    y <<= 1
                dividend -= x
                ans += y
        return ans if sign == 1 else -ans


    # 31. Next Permutation
    def nextPermutation(self, nums: List[int]) -> None:
        i = len(nums) - 2
        while i >= 0 and nums[i] >= nums[i + 1]:
            i -= 1
        if i >= 0:
            j = len(nums) - 1
            while j >= 0 and nums[j] <= nums[i]:
                j -= 1
            nums[i], nums[j] = nums[j], nums[i]

        l, r = i + 1, len(nums) - 1
        while l < r:
            nums[l], nums[r] = nums[r], nums[l]
            l += 1
            r -= 1


    # 32. Longest Valid Parentheses
    def longestValidParentheses(self, s: str) -> int:
        ans = 0
        stack = [-1]
        for i, char in enumerate(s):
            if char == "(":
                stack.append(i)
            else:
                stack.pop()
                if not stack:
                    stack.append(i)
                ans = max(ans, i - stack[-1])
        return ans


    # 33. Search in Rotated Sorted Array
    def search(self, nums: List[int], target: int) -> int:
        l, r = 0, len(nums) - 1
        while l <= r:
            mid = l + (r - l) // 2
            if nums[mid] == target:
                return mid
            elif nums[mid] <= nums[r]:
                if nums[mid] < target <= nums[r]:
                    l = mid + 1
                else:
                    r = mid - 1
            elif nums[mid] > nums[r]:
                if nums[l] <= target < nums[mid]:
                    r = mid - 1
                else:
                    l = mid + 1
        return -1


    # 34. Find First and Last Position of Element in Sorted Array
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        def first():
            i, j = 0, len(nums)
            while i < j:
                mid = i + (j - i) // 2
                if nums[mid] < target:
                    i = mid + 1
                elif nums[mid] > target:
                    j = mid
                else:
                    j = mid
            return i if i < len(nums) and nums[i] == target else -1

        def last():
            i, j = 0, len(nums)
            while i < j:
                mid = i + (j - i) // 2
                if nums[mid] < target:
                    i = mid + 1
                elif nums[mid] > target:
                    j = mid
                else:
                    i = mid + 1
            return i - 1 if i > 0 and nums[i - 1] == target else -1

        return [first(), last()]


    # 35. Search Insert Position
    def searchInsert(self, nums: List[int], target: int) -> int:
        for i, num in enumerate(nums):
            if num == target:
                return i
            elif num > target:
                break
        if num < target:
            return i + 1
        else:
            return i


    # 36. Valid Sudoku
    def isValidSudoku(self, board: List[List[str]]) -> bool:
        row = [set() for _ in range(9)]
        col = [set() for _ in range(9)]
        box = [set() for _ in range(9)]
        for i in range(9):
            for j in range(9):
                val = board[i][j]
                if val.isnumeric():
                    if (
                        val in row[i]
                        or val in col[j]
                        or val in box[(i // 3) * 3 + j // 3]
                    ):
                        return False
                    row[i].add(val)
                    col[j].add(val)
                    box[(i // 3) * 3 + j // 3].add(val)
        return True


    # 38. Count and Say
    def countAndSay(self, n: int) -> str:
        count = 1
        prev = "1"
        for j in xrange(1, n):
            res = ""
            count = 1
            for i in xrange(len(prev)):
                if i > 0:
                    if prev[i] == prev[i - 1]:
                        count += 1
                    else:
                        res += str(count)
                        res += str(prev[i - 1])
                        count = 1
                if i == len(prev) - 1:
                    res += str(count)
                    res += prev[i]
            prev = res
        return prev


    # 39. Combination Sum
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        ans = []
        n = len(candidates)
        def backtrack(idx, path_sum, path):
            if path_sum > target:
                return
            if path_sum == target:
                ans.append(path[:])
                return
            for i in range(idx, n):
                path.append(candidates[i])
                backtrack(
                    i, path_sum + candidates[i], path
                )  # because the same number can be used multiple times
                path.pop()
        backtrack(0, 0, [])
        return ans


    # 40. Combination Sum II
    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
        candidates.sort()
        def backtrack(idx, path_sum, path):
            if path_sum > target:
                return
            elif path_sum == target:
                res.append(path[:])
                return
            for i in range(idx, len(candidates)):
                if i > idx and candidates[i] == candidates[i - 1]:  # note i > idx
                    continue
                path.append(candidates[i])
                backtrack(i + 1, path_sum + candidates[i], path)
                path.pop()
        res = []
        backtrack(0, 0, [])
        return res


    # 41. First Missing Positive
    def firstMissingPositive(self, nums: List[int]) -> int:
        n = len(nums)
        for num in nums:
            if num == 1:
                break
        else:
            return 1
        for i in range(n):
            if nums[i] <= 0 or nums[i] > n:
                nums[i] = 1
        for i in range(n):
            index = abs(nums[i]) - 1
            if nums[index] > 0:
                nums[index] *= -1
        for i in range(n):
            if nums[i] > 0:
                return i + 1
        return n + 1


    # 42. Trapping Rain Water
    def trap(self, height: List[int]) -> int:
        left_max = []
        right_max = []
        cur_max = 0
        for h in height:
            cur_max = max(cur_max, h)
            left_max.append(cur_max)
        cur_max = 0
        for h in reversed(height):
            cur_max = max(cur_max, h)
            right_max.append(cur_max)
        ans = 0
        for i in range(len(height)):
            ans += min(left_max[i], right_max[-i - 1]) - height[i]
        return ans


    # 43. Multiply Strings
    def multiply(self, num1: str, num2: str) -> str:
        n1, n2 = len(num1), len(num2)
        ans = [0] * (n1 + n2)
        for i, d1 in enumerate(reversed(num1)):
            for j, d2 in enumerate(reversed(num2)):
                d1, d2 = int(d1), int(d2)
                ans[i + j] += d1 * d2
                ans[i + j + 1] += ans[i + j] // 10
                ans[i + j] %= 10
        while len(ans) > 1 and ans[-1] == 0:
            ans.pop()
        return "".join(map(str, ans[::-1]))


    # 44. Wildcard Matching
    def isMatch(self, s: str, p: str) -> bool:
        m, n = len(s), len(p)
        dp = [[False for _ in range(n + 1)] for _ in range(m + 1)]
        dp[0][0] = True
        for j in range(1, n + 1):
            dp[0][j] = dp[0][j - 1] and p[j - 1] == "*"
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if p[j - 1] == "*":
                    dp[i][j] = dp[i - 1][j] or dp[i][j - 1]
                else:
                    dp[i][j] = dp[i - 1][j - 1] and (
                        p[j - 1] == "?" or p[j - 1] == s[i - 1]
                    )
        return dp[m][n]


    # 45. Jump Game II
    def jump(self, nums: List[int]) -> int:
        n = len(nums)
        if n < 2:
            return 0
        ans = 1
        l, r = 0, nums[0]
        while r < n - 1:
            next = max([i + nums[i] for i in range(l + 1, r + 1)])
            l, r = r, next
            ans += 1
        return ans


    # 46. Permutations
    def permute(self, nums: List[int]) -> List[List[int]]:
        def dfs(nums, path):
            if len(path) == n:
                ans.append(path[:])
                return
            for i in range(len(nums)):
                dfs(nums[:i] + nums[i + 1 :], path + [nums[i]])
        ans = []
        n = len(nums)
        dfs(nums, [])
        return ans


    # 47. Permutations II
    def permuteUnique(self, nums: List[int]) -> List[List[int]]:
        def dfs(nums, path):
            if not nums:
                ans.append(path[:])
                return
            for i in range(len(nums)):
                if i > 0 and nums[i] == nums[i - 1]:
                    continue
                dfs(nums[:i] + nums[i + 1 :], path + [nums[i]])
        ans = []
        nums.sort()
        n = len(nums)
        dfs(nums, [])
        return ans


    # 48. Rotate Image
    def rotate(self, matrix: List[List[int]]) -> None:
        n = len(matrix)
        for i in range(n // 2):
            for j in range(n - n // 2):
                (
                    matrix[i][j],
                    matrix[j][-i - 1],
                    matrix[-i - 1][-j - 1],
                    matrix[-j - 1][i],
                ) = (
                    matrix[-j - 1][i],
                    matrix[i][j],
                    matrix[j][-i - 1],
                    matrix[-i - 1][-j - 1],
                )


    # 49. Group Anagrams
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        res = {}
        for x in strs:
            x_sorted = "".join(sorted(x))
            if x_sorted not in res:
                res[x_sorted] = [x]
            else:
                res[x_sorted].append(x)
        return res.values()


    # 50. Pow(x, n)
    def myPow(self, x: float, n: int) -> float:
        if n == 0:
            return 1
        if n < 0:
            return 1 / self.myPow(x, -n)
        tmp = self.myPow(x, n // 2)
        if n % 2 == 0:
            return tmp * tmp
        else:
            return tmp * tmp * x


    # 51. N-Queens
    def solveNQueens(self, n: int) -> List[List[str]]:
        if n < 1:
            return []
        states = []
        cols = set()
        pie = set()
        na = set()
        def backtrack(row, path):
            if row == n:
                states.append(path[:])
                return
            for col in range(n):
                if col in cols or row + col in pie or row - col in na:
                    continue
                cols.add(col)
                pie.add(row + col)
                na.add(row - col)
                path.append(col)
                backtrack(row + 1, path)
                path.pop()
                cols.remove(col)
                pie.remove(row + col)
                na.remove(row - col)
        backtrack(0, [])
        ans = []
        for state in states:
            board = []
            for col in state:
                board.append("." * col + "Q" + "." * (n - col - 1))
            ans.append(board)
        return ans


    # 53. Maximum Subarray
    def maxSubArray(self, nums: List[int]) -> int:
        n = len(nums)
        curr_sum = max_sum = nums[0]
        for i in range(1, n):
            curr_sum = max(nums[i], curr_sum + nums[i])
            max_sum = max(max_sum, curr_sum)
        return max_sum


    # 54. Spiral Matrix
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        directions = [[0, 1], [1, 0], [0, -1], [-1, 0]]
        x = y = d = 0
        ans = []
        visited = set()
        m, n = len(matrix), len(matrix[0])
        for _ in range(m * n):
            ans.append(matrix[x][y])
            visited.add((x, y))
            next_x, next_y = x + directions[d][0], y + directions[d][1]
            if (
                next_x < 0
                or next_x >= m
                or next_y < 0
                or next_y >= n
                or (next_x, next_y) in visited
            ):
                d = (d + 1) % 4
                x, y = x + directions[d][0], y + directions[d][1]
            else:
                x, y = next_x, next_y
        return ans


    # 55. Jump Game
    def canJump(self, nums: List[int]) -> bool:
        furthest = 0
        for i in range(len(nums)):
            if i <= furthest:
                furthest = max(furthest, nums[i] + i)
                if furthest >= len(nums) - 1:
                    return True
            else:
                return False
        return True


    # 56. Merge Intervals
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        intervals.sort()
        ans = []
        for s, e in intervals:
            if not ans or ans[-1][1] < s:
                ans.append([s, e])
            else:
                ans[-1][1] = max(ans[-1][1], e)
        return ans


    # 57. Insert Interval
    def insert(
        self, intervals: List[List[int]], newInterval: List[int]
    ) -> List[List[int]]:
        i, n = 0, len(intervals)
        res = []
        while i < n and newInterval[0] > intervals[i][0]:
            res.append(intervals[i])
            i += 1
        if not res or res[-1][1] < newInterval[0]:  # no overlap
            res.append(newInterval)
        else:
            res[-1][1] = max(res[-1][1], newInterval[1])
        while i < n:
            if res[-1][1] < intervals[i][0]:
                res.append(intervals[i])
            else:
                res[-1][1] = max(res[-1][1], intervals[i][1])
            i += 1
        return res


    # 58. Length of Last Word
    def lengthOfLastWord(self, s: str) -> int:
        words = s.split(" ")[::-1]
        for word in words:
            if word != "":
                return len(word)
        return 0


    # 62. Unique Paths
    def uniquePaths(self, m: int, n: int) -> int:
        dp = [[0] * n for _ in range(m)]
        for i in range(m):
            dp[i][0] = 1
        for j in range(n):
            dp[0][j] = 1
        for i in range(1, m):
            for j in range(1, n):
                dp[i][j] = dp[i][j - 1] + dp[i - 1][j]
        return dp[m - 1][n - 1]


    # 64. Minimum Path Sum
    def minPathSum(self, grid: List[List[int]]) -> int:
        row = len(grid)
        col = len(grid[0])
        dist = [[0 for i in range(col)] for i in range(row)]
        dist[0][0] = grid[0][0]
        for i in range(1, row):
            dist[i][0] = dist[i - 1][0] + grid[i][0]
        for j in range(1, col):
            dist[0][j] = dist[0][j - 1] + grid[0][j]
        for i in range(1, row):
            for j in range(1, col):
                dist[i][j] = min(dist[i - 1][j], dist[i][j - 1]) + grid[i][j]
        return dist[row - 1][col - 1]


    # 65. Valid Number
    def isNumber(self, s: str) -> bool:
        s = s.strip()
        has_dot = has_e = has_digit = False
        for i, char in enumerate(s):
            if char == "e":
                if has_e or not has_digit:
                    return False
                has_e = True
                has_digit = False
            elif char in "-+":
                if i > 0 and s[i - 1] != "e":
                    return False
            elif char == ".":
                if has_e or has_dot:
                    return False
                has_dot = True
            elif char.isdecimal():
                has_digit = True
            else:
                return False
        return has_digit


    # 66. Plus One
    def plusOne(self, digits: List[int]) -> List[int]:
        carry = 1
        for i in range(len(digits) - 1, -1, -1):
            tmp = digits[i] + carry
            digits[i] = tmp % 10
            carry = tmp // 10
        if carry > 0:
            digits = [1] + digits
        return digits


    # 67. Add Binary
    def addBinary(self, a: str, b: str) -> str:
        n1, n2 = len(a), len(b)
        i, j = n1 - 1, n2 - 1
        carry = 0
        ans = []
        while i >= 0 and j >= 0:
            tmp = carry + int(a[i]) + int(b[j])
            carry = tmp // 2
            ans.append(str(tmp % 2))
            i -= 1
            j -= 1
        while i >= 0:
            tmp = carry + int(a[i])
            carry = tmp // 2
            ans.append(str(tmp % 2))
            i -= 1
        while j >= 0:
            tmp = carry + int(b[j])
            carry = tmp // 2
            ans.append(str(tmp % 2))
            j -= 1
        if carry:
            ans.append(str(carry))
        return "".join(ans[::-1])


    # 69. Sqrt(x)
    def mySqrt(self, x: int) -> int:
        if x < 2:
            return x
        x0 = x
        x1 = (x0 - x / x0) / 2
        while abs(x1 - x0) >= 1:
            x0 = x1
            x1 = (x0 + x / x0) / 2
        return int(x1)


    # 70. Climbing Stairs
    def climbStairs(self, n: int) -> int:
        if n <= 2:
            return n
        dp = [0] * (n + 1)
        dp[1] = 1
        dp[2] = 2
        for i in range(3, n + 1):
            dp[i] = dp[i - 1] + dp[i - 2]
        return dp[n]


    # 71. Simplify Path
    def simplifyPath(self, path: str) -> str:
        if not str:
            return str
        stack = []
        for portion in path.split("/"):
            if portion == "." or portion == "":
                continue
            elif portion == "..":
                if stack:
                    stack.pop()
            else:
                stack.append(portion)
        return "/" + "/".join(stack)


    # 72. Edit Distance
    def minDistance(self, word1: str, word2: str) -> int:
        n1, n2 = len(word1), len(word2)
        if n1 * n2 == 0:
            return max(n1, n2)
        dp = [[0] * (n2 + 1) for _ in range(n1 + 1)]
        for i in range(n1 + 1):
            dp[i][0] = i
        for j in range(n2 + 1):
            dp[0][j] = j
        for i in range(1, n1 + 1):
            for j in range(1, n2 + 1):
                if word1[i - 1] == word2[j - 1]:
                    dp[i][j] = min(dp[i - 1][j - 1], dp[i - 1][j] + 1, dp[i][j - 1] + 1)
                else:
                    dp[i][j] = min(dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1]) + 1
        return dp[n1][n2]


    # 73. Set Matrix Zeroes
    def setZeroes(self, matrix: List[List[int]]) -> None:
        m, n = len(matrix), len(matrix[0])
        first_row = first_col = False
        for i in range(m):
            if matrix[i][0] == 0:
                first_col = True
                break
        for j in range(n):
            if matrix[0][j] == 0:
                first_row = True
                break
        for i in range(1, m):
            for j in range(1, n):
                if matrix[i][j] == 0:
                    matrix[i][0] = matrix[0][j] = 0
        for i in range(1, m):
            if matrix[i][0] == 0:
                for j in range(1, n):
                    matrix[i][j] = 0
        for j in range(1, n):
            if matrix[0][j] == 0:
                for i in range(1, m):
                    matrix[i][j] = 0
        if first_row:
            for j in range(n):
                matrix[0][j] = 0
        if first_col:
            for i in range(m):
                matrix[i][0] = 0


    # 74. Search a 2D Matrix
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        if not matrix:
            return False
        m, n = len(matrix), len(matrix[0])
        left, right = 0, m * n - 1
        while left <= right:
            mid = left + (right - left) // 2
            x, y = mid // n, mid % n
            if target == matrix[x][y]:
                return True
            elif target < matrix[x][y]:
                right = mid - 1
            else:
                left = mid + 1
        return False


    # 75. Sort Colors
    def sortColors(self, nums: List[int]) -> None:
        p0 = i = 0
        p2 = len(nums) - 1
        while i <= p2:
            if nums[i] == 0:
                nums[i], nums[p0] = nums[p0], nums[i]
                p0 += 1
                i += 1
            elif nums[i] == 2:
                nums[i], nums[p2] = nums[p2], nums[i]
                p2 -= 1
            else:
                i += 1


    # 76. Minimum Window Substring
    def minWindow(self, s: str, t: str) -> str:
        t_counter = collections.Counter(t)
        s_counter = collections.defaultdict(int)
        i = matched = 0
        min_len = float("inf")
        pos = []
        for j in range(len(s)):
            if s[j] in t_counter:
                s_counter[s[j]] += 1
                if s_counter[s[j]] == t_counter[s[j]]:
                    matched += 1
            while i <= j and matched == len(t_counter):
                if j - i + 1 < min_len:
                    min_len = j - i + 1
                    pos = [i, j]
                if s[i] in t_counter:
                    s_counter[s[i]] -= 1
                    if s_counter[s[i]] < t_counter[s[i]]:
                        matched -= 1
                i += 1
        return s[pos[0] : pos[1] + 1] if pos else ""


    # 78. Subsets
    def subsets(self, nums: List[int]) -> List[List[int]]:
        def backtrack(idx, path):
            ans.append(path[:])
            for i in range(idx, n):
                path.append(nums[i])
                backtrack(i + 1, path)
                path.pop()
        ans = []
        n = len(nums)
        backtrack(0, [])
        return ans


    # 79. Word Search
    def exist(self, board: List[List[str]], word: str) -> bool:
        m, n = len(board), len(board[0])
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        def backtrack(i, j, word):
            if not word:
                return True
            if 0 <= i < m and 0 <= j < n and board[i][j] == word[0]:
                board[i][j] = "*"
                for d in directions:
                    next_i, next_j = i + d[0], j + d[1]
                    if backtrack(next_i, next_j, word[1:]):
                        return True
                board[i][j] = word[0]
            return False
        for i in range(m):
            for j in range(n):
                if backtrack(i, j, word):
                    return True
        return False


    # 80. Remove Duplicates from Sorted Array II
    def removeDuplicates(self, nums: List[int]) -> int:
        j = count = 1
        for i in range(1, len(nums)):
            if nums[i] == nums[i - 1]:
                count += 1
            else:
                count = 1
            if count <= 2:
                nums[j] = nums[i]
                j += 1
        return j


    # 81. Search in Rotated Sorted Array II
    def search(self, nums: List[int], target: int) -> bool:
        left, right = 0, len(nums) - 1
        while left <= right:
            mid = left + (right - left) // 2
            if nums[mid] == target:
                return True
            elif nums[mid] < nums[right]:
                if nums[mid] < target <= nums[right]:
                    left = mid + 1
                else:
                    right = mid - 1
            elif nums[mid] > nums[right]:
                if nums[left] <= target < nums[mid]:
                    right = mid - 1
                else:
                    left = mid + 1
            else:
                right -= 1
        return False


    # 83. Remove Duplicates from Sorted List
    def deleteDuplicates(self, head: ListNode) -> ListNode:
        dummy = ListNode(float("-Inf"))
        dummy.next = head
        p = dummy
        while p.next:
            if p.val == p.next.val:
                p.next = p.next.next
            else:
                p = p.next
        return dummy.next


    # 84. Largest Rectangle in Histogram
    def largestRectangleArea(self, heights: List[int]) -> int:
        stack = [-1]
        heights.append(0)
        ans = 0
        for i, h in enumerate(heights):
            while heights[stack[-1]] > h:
                idx = stack.pop()
                ans = max(ans, heights[idx] * (i - 1 - stack[-1]))
            stack.append(i)
        return ans


    # 88. Merge Sorted Array
    def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        p1 = m - 1
        p2 = n - 1
        p = m + n - 1
        while p >= 0 and p > p1 and p1 >= 0 and p2 >= 0:
            if nums1[p1] >= nums2[p2]:
                nums1[p] = nums1[p1]
                p = p - 1
                p1 = p1 - 1
            else:
                nums1[p] = nums2[p2]
                p = p - 1
                p2 = p2 - 1
        if p2 >= 0:
            nums1[0 : (p + 1)] = nums2[: p2 + 1]
        return nums1


    # 90. Subsets II
    def subsetsWithDup(self, nums: List[int]) -> List[List[int]]:
        def backtrack(idx, path):
            ans.append(path[:])
            for i in range(idx, n):
                if i == idx or nums[i] != nums[i - 1]:
                    path.append(nums[i])
                    backtrack(i + 1, path)
                    path.pop()
        ans = []
        n = len(nums)
        nums.sort()
        backtrack(0, [])
        return ans


    # 91. Decode Ways
    def numDecodings(self, s: str) -> int:
        n = len(s)
        memo = [None] * (n + 1)
        memo[0] = 1
        def dp(n):
            if memo[n] is not None:
                return memo[n]
            ans = 0
            if n >= 1 and s[n - 1] != "0":
                ans += dp(n - 1)
            if n >= 2 and "10" <= s[n - 2 : n] <= "26":
                ans += dp(n - 2)
            memo[n] = ans
            return ans
        return dp(n)


    # 92. Reverse Linked List II
    def reverseBetween(self, head: ListNode, m: int, n: int) -> ListNode:
        if not head:
            return
        prev, cur = None, head
        while m > 1:
            m -= 1
            n -= 1
            prev = cur
            cur = cur.next
        node1, node2 = prev, cur
        while n > 0:
            next = cur.next
            cur.next = prev
            prev = cur
            cur = next
            n -= 1
        if node1:
            node1.next = prev
        else:
            head = prev
        node2.next = cur
        return head


    # 94. Binary Tree Inorder Traversal
    def inorderTraversal(self, root: TreeNode) -> List[int]:
        def helper(root, ans):
            if not root:
                return
            if root.left:
                helper(root.left, ans)
            ans.append(root.val)
            if root.right:
                helper(root.right, ans)
        ans = []
        helper(root, ans)
        return ans


    # 98. Validate Binary Search Tree
    def isValidBST(self, root: TreeNode) -> bool:
        def helper(root, lo, hi):
            if not root:
                return True
            if lo < root.val < hi:
                return helper(root.left, lo, root.val) and helper(
                    root.right, root.val, hi
                )
        return helper(root, float("-inf"), float("inf"))


    # 100. Same Tree
    def isSameTree(self, p: TreeNode, q: TreeNode) -> bool:
        if not p and not q:
            return True
        if (not p and q) or (p and not q):
            return False
        if p.val == q.val:
            return self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right)


    # 101. Symmetric Tree
    def isSymmetric(self, root: TreeNode) -> bool:
        def helper(node1, node2):
            if not node1 and not node2:
                return True
            if (not node1 and node2) or (node1 and not node2):
                return False
            return (
                node1.val == node2.val
                and helper(node1.left, node2.right)
                and helper(node1.right, node2.left)
            )
        if not root:
            return True
        return helper(root.left, root.right)


    # 102. Binary Tree Level Order Traversal
    def levelOrder(self, root: TreeNode) -> List[List[int]]:
        if not root:
            return []
        q = deque()
        q.append(root)
        ans = []
        while q:
            level = []
            n = len(q)
            for _ in range(n):
                node = q.popleft()
                level.append(node.val)
                if node.left:
                    q.append(node.left)
                if node.right:
                    q.append(node.right)
            ans.append(level)
        return ans


    # 103. Binary Tree Zigzag Level Order Traversal
    def zigzagLevelOrder(self, root: TreeNode) -> List[List[int]]:
        if not root:
            return []
        res = []
        q = deque()
        q.append(root)
        left_to_right = True
        while q:
            n = len(q)
            level = []
            for _ in range(n):
                if left_to_right:
                    node = q.popleft()
                    level.append(node.val)
                    if node.left:
                        q.append(node.left)
                    if node.right:
                        q.append(node.right)
                else:
                    node = q.pop()
                    level.append(node.val)
                    if node.right:
                        q.appendleft(node.right)
                    if node.left:
                        q.appendleft(node.left)
            res.append(level)
            left_to_right = False if left_to_right else True
        return res


    # 104. Maximum Depth of Binary Tree
    def maxDepth(self, root: TreeNode) -> int:
        if not root:
            return 0
        left = self.maxDepth(root.left)
        right = self.maxDepth(root.right)
        return max(left, right) + 1


    # 105. Construct Binary Tree from Preorder and Inorder Traversal
    def buildTree(self, preorder: List[int], inorder: List[int]) -> TreeNode:
        def helper(inorder_idx_left, inorder_idx_right):
            if inorder_idx_left <= inorder_idx_right:
                val = preorder[self.preorder_idx]
                root = TreeNode(val)
                inorder_idx = inorder_idx_map[val]
                self.preorder_idx += 1
                root.left = helper(inorder_idx_left, inorder_idx - 1)
                root.right = helper(inorder_idx + 1, inorder_idx_right)
                return root
        inorder_idx_map = {val: idx for idx, val in enumerate(inorder)}
        self.preorder_idx = 0
        return helper(0, len(inorder) - 1)


    # 107. Binary Tree Level Order Traversal II
    def levelOrderBottom(self, root: TreeNode) -> List[List[int]]:
        if not root:
            return []
        res, current = [], [root]
        while current:
            val, next = [], []
            for node in current:
                val.append(node.val)
                if node.left:
                    next.append(node.left)
                if node.right:
                    next.append(node.right)
            res.append(val)
            current = next
        return res[::-1]


    # 108. Convert Sorted Array to Binary Search Tree
    def sortedArrayToBST(self, nums: List[int]) -> TreeNode:
        if nums:
            mid = len(nums) // 2
            node = TreeNode(nums[mid])
            node.left = self.sortedArrayToBST(nums[:mid])
            node.right = self.sortedArrayToBST(nums[mid + 1 :])
            return node


    # 109. Convert Sorted List to Binary Search Tree
    def sortedListToBST(self, head: ListNode) -> TreeNode:
        vals = []
        while head:
            vals.append(head.val)
            head = head.next
        def helper(l, r):
            if l > r:
                return
            mid = (l + r) // 2
            node = TreeNode(vals[mid])
            if l == r:
                return node
            node.left = helper(l, mid - 1)
            node.right = helper(mid + 1, r)
            return node
        return helper(0, len(vals) - 1)


    # 110. Balanced Binary Tree
    def isBalanced(self, root: TreeNode) -> bool:
        def getHeight(root):
            if not root:
                return 0
            left_height, right_height = getHeight(root.left), getHeight(root.right)
            if (
                left_height < 0
                or right_height < 0
                or abs(left_height - right_height) > 1
            ):
                return -1
            return max(left_height, right_height) + 1
        return getHeight(root) >= 0


    # 111. Minimum Depth of Binary Tree
    def minDepth(self, root: TreeNode) -> int:
        if not root:
            return 0
        if not root.left:
            return 1 + self.minDepth(root.right)
        if not root.right:
            return 1 + self.minDepth(root.left)
        return 1 + min(self.minDepth(root.left), self.minDepth(root.right))


    # 112. Path Sum
    def hasPathSum(self, root: TreeNode, sum: int) -> bool:
        if not root:
            return False
        sum -= root.val
        if not root.left and not root.right:
            if sum == 0:
                return True
        if (root.left and self.hasPathSum(root.left, sum)) or (
            root.right and self.hasPathSum(root.right, sum)
        ):
            return True
        return False


    # 114. Flatten Binary Tree to Linked List
    def flatten(self, root: TreeNode) -> None:
        if not root:
            return
        self.flatten(root.left)
        self.flatten(root.right)
        if root.left:
            left = root.left
            while left.right:
                left = left.right
            left.right = root.right
            root.right = root.left
            root.left = None


    # 116. Populating Next Right Pointers in Each Node
    def connect(self, root: Node) -> Node:
        if not root:
            return
        q = collections.deque()
        q.append(root)
        while q:
            n = len(q)
            for i in range(n):
                node = q.popleft()
                if i != n - 1:
                    node.next = q[0]
                if node.left:
                    q.append(node.left)
                if node.right:
                    q.append(node.right)
        return root


    # 117. Populating Next Right Pointers in Each Node II
    def connect2(self, root: Node) -> Node:
        if not root:
            return
        q = collections.deque()
        q.append(root)
        while q:
            n = len(q)
            for i in range(n):
                node = q.popleft()
                if i != n - 1:
                    node.next = q[0]
                if node.left:
                    q.append(node.left)
                if node.right:
                    q.append(node.right)
        return root


    # 118. Pascal's Triangle
    def generate(self, numRows: int) -> List[List[int]]:
        res = []
        if numRows >= 1:
            res.append([1])
        if numRows >= 2:
            res.append([1, 1])
        for i in range(3, numRows + 1):
            tmp = [1]
            for j in range(i - 2):
                tmp.append(res[-1][j] + res[-1][j + 1])
            tmp.append(1)
            res.append(tmp)
        return res


    # 119. Pascal's Triangle II
    def getRow(self, rowIndex: int) -> List[int]:
        result = [1] + [0] * rowIndex
        if rowIndex > 0:
            for i in xrange(1, rowIndex + 1):
                for j in reversed(xrange(1, i + 1)):
                    result[j] += result[j - 1]
        return result


    # 121. Best Time to Buy and Sell Stock
    def maxProfit(self, prices: List[int]) -> int:
        l = 0
        ans = 0
        m = float("inf")
        for i in range(len(prices)):
            ans = max(ans, prices[i] - m)
            m = min(m, prices[i])
        return ans


    # 122. Best Time to Buy and Sell Stock II
    def maxProfit2(self, prices: List[int]) -> int:
        ret = 0
        for i in range(1, len(prices)):
            diff = prices[i] - prices[i - 1]
            if diff > 0:
                ret += diff
        return ret


    # 123. Best Time to Buy and Sell Stock III
    def maxProfit3(self, prices: List[int]) -> int:
        n = len(prices)
        if n < 2:
            return 0
        st = prices[0]
        mp = []
        mprof = 0
        for i in range(0, n):
            if prices[i] - st > mprof:
                mprof = prices[i] - st
            if prices[i] < st:
                st = prices[i]
            mp.append(mprof)
        ed = prices[-1]
        mprof = 0
        ed = prices[-1]
        for i in range(n - 1, -1, -1):
            if ed - prices[i] + mp[i] > mprof:
                mprof = ed - prices[i] + mp[i]
            if prices[i] > ed:
                ed = prices[i]
        return mprof


    # 124. Binary Tree Maximum Path Sum
    def maxPathSum(self, root: TreeNode) -> int:
        self.ans = float("-inf")
        def helper(root):
            if not root:
                return 0
            left_sum = max(helper(root.left), 0)
            right_sum = max(helper(root.right), 0)
            self.ans = max(self.ans, left_sum + root.val + right_sum)
            return max(left_sum, right_sum) + root.val
        helper(root)
        return self.ans


    # 125. Valid Palindrome
    def isPalindrome(self, s: str) -> bool:
        n = len(s)
        i, j = 0, n - 1
        while i < j:
            while i < j and not s[i].isalnum():
                i += 1
            while i < j and not s[j].isalnum():
                j -= 1
            if i < j:
                if s[i].lower() != s[j].lower():
                    return False
                i += 1
                j -= 1
        return True


    # 126. Word Ladder II
    def findLadders(
        self, beginWord: str, endWord: str, wordList: List[str]
    ) -> List[List[str]]:
        if beginWord == endWord:
            return [[endWord]]
        wordSet = set(wordList)
        if endWord not in wordSet:
            return []
        # bfs
        distance = 0
        graph = collections.defaultdict(set)
        visited = set([beginWord])
        cur = [beginWord]
        found = False
        while cur and not found:
            next = set()
            for word in cur:
                n = len(word)
                for i in range(n):
                    for ch in string.ascii_lowercase:
                        if word[i] != ch:
                            new_word = word[:i] + ch + word[i + 1 :]
                            if new_word in wordSet and new_word not in visited:
                                graph[word].add(new_word)
                                next.add(new_word)
                            if new_word == endWord:
                                found = True
            cur = next
            visited = visited.union(next)
        # dfs
        if not found:
            return []
        ans = []
        def backtrack(word, path):
            if word == endWord:
                ans.append(path[:])
                return
            for nbr in graph[word]:
                path.append(nbr)
                backtrack(nbr, path)
                path.pop()

        backtrack(beginWord, [beginWord])
        return ans


    # 127. Word Ladder
    def ladderLength(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
        distance = 0
        curr = [beginWord]
        visited = set([beginWord])
        wordSet = set(wordList)
        while curr:
            next = []
            for word in curr:
                if word == endWord:
                    return distance + 1
                for i in range(len(word)):
                    for c in ascii_lowercase:
                        if c != word[i]:
                            candidate = word[:i] + c + word[i + 1 :]
                            if candidate not in visited and candidate in wordSet:
                                next.append(candidate)
                                visited.add(candidate)
            curr = next
            distance += 1
        return 0


    # 128. Longest Consecutive Sequence
    def longestConsecutive(self, nums: List[int]) -> int:
        nums_set = set(nums)
        ans = 0
        for num in nums:
            if num - 1 not in nums_set:
                cur_ans = 1
                while num + 1 in nums_set:
                    cur_ans += 1
                    num += 1
                ans = max(ans, cur_ans)
        return ans


    # 129. Sum Root to Leaf Numbers
    def sumNumbers(self, root: TreeNode) -> int:
        if not root:
            return 0
        cur = [(root, 0)]
        ans = 0
        while cur:
            next = []
            for node, prev_val in cur:
                cur_val = 10 * prev_val + node.val
                if not node.left and not node.right:
                    ans += cur_val
                if node.left:
                    next.append([node.left, cur_val])
                if node.right:
                    next.append([node.right, cur_val])
            cur = next
        return ans


    # 130. Surrounded Regions
    def solve(self, board: List[List[str]]) -> None:
        if not board:
            return
        m, n = len(board), len(board[0])
        directions = [[-1, 0], [1, 0], [0, 1], [0, -1]]
        q = collections.deque()
        for i in range(m):
            for j in range(n):
                if ((i == 0 or i == m - 1) or (j == 0 or j == n - 1)) and board[i][
                    j
                ] == "O":
                    q.append((i, j))
        while q:
            i, j = q.popleft()
            board[i][j] = "E"
            for direction in directions:
                next_i, next_j = i + direction[0], j + direction[1]
                if 0 <= next_i < m and 0 <= next_j < n and board[next_i][next_j] == "O":
                    q.append((next_i, next_j))
        for i in range(m):
            for j in range(n):
                if board[i][j] == "E":
                    board[i][j] = "O"
                elif board[i][j] == "O":
                    board[i][j] = "X"

    # 131. Palindrome Partitioning
    def partition(self, s: str) -> List[List[str]]:
        def isPalindrome(s):
            return s == s[::-1]
        def backtrack(idx, path):
            if idx == n:
                ans.append(path[:])
                return
            for i in range(idx, n):
                if isPalindrome(s[idx : i + 1]):
                    path.append(s[idx : i + 1])
                    backtrack(i + 1, path)
                    path.pop()
        n = len(s)
        ans = []
        backtrack(0, [])
        return ans


    # 133. Clone Graph
    def cloneGraph(self, node: Node) -> Node:
        if not node:
            return
        q = collections.deque()
        q.append(node)
        visited = {}
        visited[node] = Node(node.val)
        while q:
            n = q.popleft()
            for nbr in n.neighbors:
                if nbr not in visited:
                    q.append(nbr)
                    visited[nbr] = Node(nbr.val)
                visited[n].neighbors.append(visited[nbr])
        return visited[node]


    # 134. Gas Station
    def canCompleteCircuit(self, gas: List[int], cost: List[int]) -> int:
        if sum(gas) < sum(cost):
            return -1
        tank = start = 0
        for i in range(len(gas)):
            tank += gas[i] - cost[i]
            if tank < 0:
                tank = 0
                start = i + 1
        return start


    # 136. Single Number
    def singleNumber(self, nums: List[int]) -> int:
        res = 0
        for num in nums:
            res ^= num
        return res


    # 137. Single Number II
    def singleNumber2(self, nums: List[int]) -> int:
        return [x for x, val in collections.Counter(nums).items() if val == 1][0]


    # 138. Copy List with Random Pointer
    def copyRandomList(self, head: "Node") -> "Node":
        dummy = ListNode()
        prev = dummy
        node = head
        map = {}
        while node:
            prev.next = Node(node.val)
            map[node] = prev.next
            node = node.next
            prev = prev.next
        prev = dummy.next
        node = head
        while prev:
            if node.random:
                prev.random = map[node.random]
            prev = prev.next
            node = node.next
        return dummy.next


    # 139. Word Break
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        n = len(s)
        wordSet = set(wordDict)
        dp = [False] * (n + 1)
        dp[0] = True
        for i in range(1, n + 1):
            for j in range(i):
                if dp[j] and s[j:i] in wordSet:
                    dp[i] = True
                    break
        return dp[-1]


    # 140. Word Break II
    def wordBreak2(self, s: str, wordDict: List[str]) -> List[str]:
        wordSet = set(wordDict)
        n = len(s)
        dp = [False] * (n + 1)
        dp[0] = True
        for i in range(n + 1):
            for j in range(i):
                if dp[j] and s[j:i] in wordSet:
                    dp[i] = True
                    break
        if not dp[-1]:
            return []
        ans = []
        def backtrack(idx, path):
            if idx == n:
                ans.append(" ".join(path))
                return
            for i in range(idx, n):
                if s[idx : i + 1] in wordSet:
                    path.append(s[idx : i + 1])
                    backtrack(i + 1, path)
                    path.pop()
        backtrack(0, [])
        return ans


    # 141. Linked List Cycle
    def hasCycle(self, head: ListNode) -> bool:
        slow = fast = head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            if slow == fast:
                return True
        return False


    # 143. Reorder List
    def reorderList(self, head: ListNode) -> None:
        def findMid(head):
            slow, fast = head, head
            while fast and fast.next:
                slow = slow.next
                fast = fast.next.next
            return slow
        def reverse(head):
            prev, cur = None, head
            while cur:
                next = cur.next
                cur.next = prev
                prev = cur
                cur = next
            return prev
        def merge(node1, node2):
            head = node1  # assume node1 is always the longer one
            while node1 and node2:
                next1 = node1.next
                next2 = node2.next
                node1.next = node2
                node2.next = next1
                node1 = next1
                node2 = next2
            return head
        if not head:
            return
        mid = findMid(head)
        next = mid.next
        mid.next = None
        end = reverse(next)
        head = merge(head, end)
        return head


    # 145. Binary Tree Postorder Traversal
    def postorderTraversal(self, root: TreeNode) -> List[int]:
        ans = []
        def helper(root):
            if not root:
                return
            helper(root.left)
            helper(root.right)
            ans.append(root.val)
        helper(root)
        return ans


    # 148. Sort List
    def sortList(self, head: ListNode) -> ListNode:
        def findMid(head):
            slow, fast = head, head
            while fast.next and fast.next.next:
                slow = slow.next
                fast = fast.next.next
            return slow
        def merge(head1, head2):
            prev = dummy = ListNode()
            while head1 and head2:
                if head1.val < head2.val:
                    prev.next = head1
                    head1 = head1.next
                    prev = prev.next
                else:
                    prev.next = head2
                    head2 = head2.next
                    prev = prev.next
            if head1:
                prev.next = head1
            if head2:
                prev.next = head2
            return dummy.next
        def helper(head):
            if not head or not head.next:
                return head
            mid = findMid(head)
            head1 = head
            head2 = mid.next
            mid.next = None
            head1 = helper(head1)
            head2 = helper(head2)
            return merge(head1, head2)
        return helper(head)


    # 150. Evaluate Reverse Polish Notation
    def evalRPN(self, tokens: List[str]) -> int:
        stack = []
        for token in tokens:
            if token in "+-*/":
                x = int(stack.pop())
                y = int(stack.pop())
                if token == "+":
                    stack.append(y + x)
                elif token == "-":
                    stack.append(y - x)
                elif token == "*":
                    stack.append(y * x)
                else:
                    stack.append(int(y / x))
            else:
                stack.append(token)
        return stack[0]


    # 151. Reverse Words in a String
    def reverseWords(self, s: str) -> str:
        words = [word for word in s.split(" ") if word not in ("", " ")]
        return " ".join(words[::-1])


    # 152. Maximum Product Subarray
    def maxProduct(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 0:
            return
        if n == 1:
            return nums[0]
        max_so_far = min_so_far = ans = nums[0]
        for i in range(1, n):
            prev_max_so_far = max_so_far
            max_so_far = max(nums[i], prev_max_so_far * nums[i], min_so_far * nums[i])
            min_so_far = min(nums[i], prev_max_so_far * nums[i], min_so_far * nums[i])
            ans = max(ans, max_so_far)
        return ans


    # 153. Find Minimum in Rotated Sorted Array
    def findMin(self, nums: List[int]) -> int:
        left, right = 0, len(nums) - 1
        while left < right:
            mid = left + (right - left) // 2
            if nums[mid] > nums[right]:
                left = mid + 1
            else:
                right = mid
        return nums[left]


    # 154. Find Minimum in Rotated Sorted Array II
    def findMin(self, nums: List[int]) -> int:
        l, r = 0, len(nums) - 1
        while l < r:
            mid = l + (r - l) // 2
            if nums[mid] > nums[r]:
                l = mid + 1
            elif nums[mid] < nums[r]:
                r = mid
            else:
                r -= 1
        return nums[l]


    # 160. Intersection of Two Linked Lists
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> ListNode:
        node1 = headA
        node2 = headB
        while node1 and node2:
            node1 = node1.next
            node2 = node2.next
        if not node2:
            node2 = headA
        else:
            node1 = headB
        while node1 and node2:
            node1 = node1.next
            node2 = node2.next
        if not node2:
            node2 = headA
        else:
            node1 = headB
        while node1 and node2:
            if node1 == node2:
                return node1
            node1 = node1.next
            node2 = node2.next


    # 162. Find Peak Element
    def findPeakElement(self, nums: List[int]) -> int:
        n = len(nums)
        l, r = 0, n - 1
        while l <= r:
            mid = l + (r - l) // 2
            if (mid == 0 or nums[mid - 1] < nums[mid]) and (
                mid == n - 1 or nums[mid] > nums[mid + 1]
            ):
                return mid
            elif mid > 0 and nums[mid - 1] >= nums[mid]:
                r = mid - 1
            elif mid < n - 1 and nums[mid] <= nums[mid + 1]:
                l = mid + 1


    # 165. Compare Version Numbers
    def findPeakElement(self, nums: List[int]) -> int:
        n = len(nums)
        l, r = 0, n - 1
        while l <= r:
            mid = l + (r - l) // 2
            if (mid == 0 or nums[mid - 1] < nums[mid]) and (
                mid == n - 1 or nums[mid] > nums[mid + 1]
            ):
                return mid
            elif mid > 0 and nums[mid - 1] >= nums[mid]:
                r = mid - 1
            elif mid < n - 1 and nums[mid] <= nums[mid + 1]:
                l = mid + 1


    # 166. Fraction to Recurring Decimal
    def fractionToDecimal(self, numerator: int, denominator: int) -> str:
        if numerator % denominator == 0:
            return str(numerator // denominator)
        sign = "-" if (numerator < 0) ^ (denominator < 0) else ""
        numerator, denominator = abs(numerator), abs(denominator)
        ans = sign + str(numerator // denominator) + "."
        numerator %= denominator
        i = 0
        decimals = ""
        map = {numerator: i}
        while numerator:
            numerator *= 10
            i += 1
            decimals += str(numerator // denominator)
            numerator %= denominator
            if numerator in map:
                decimals = (
                    decimals[: map[numerator]] + "(" + decimals[map[numerator] :] + ")"
                )
                break
            map[numerator] = i
        return ans + decimals


    # 167. Two Sum II - Input array is sorted
    def twoSum2(self, numbers: List[int], target: int) -> List[int]:
        i, j = 0, len(numbers) - 1
        res = []
        while i < j:
            sum = numbers[i] + numbers[j]
            if sum < target:
                i += 1
            elif sum > target:
                j -= 1
            else:
                return [i + 1, j + 1]


    # 168. Excel Sheet Column Title
    def convertToTitle(self, n: int) -> str:
        res = ""
        while n:
            res = chr(ord("A") + (n - 1) % 26) + res
            n = (n - 1) / 26
        return res


    # 169. Majority Element
    def majorityElement(self, nums: List[int]) -> int:
        res = None
        count = 0
        for num in nums:
            if count == 0:
                res = num
            if num == res:
                count += 1
            else:
                count -= 1
        return res


    # 171. Excel Sheet Column Number
    def titleToNumber(self, s: str) -> int:
        return sum(
            [(ord(char) - ord("A") + 1) * 26 ** i for i, char in enumerate(reversed(s))]
        )


    # 172. Factorial Trailing Zeroes
    def trailingZeroes(self, n: int) -> int:
        ans = 0
        while n > 0:
            n //= 5
            ans += n
        return ans

    
    # 179. Largest Number
    def largestNumber(self, nums):
        s = "".join(
            sorted(map(str, nums), key=cmp_to_key(lambda x, y: int(y + x) - int(x + y)))
        )
        return "0" if s[0] == "0" else s


    # 189. Rotate Array
    def rotate(self, nums: List[int], k: int) -> None:
        def reverse(nums, i, j):
            while i < j:
                nums[i], nums[j] = nums[j], nums[i]
                i += 1
                j -= 1
        n = len(nums)
        k = k % n
        reverse(nums, 0, n - 1)
        reverse(nums, 0, k - 1)
        reverse(nums, k, n - 1)


    # 190. Reverse Bits
    def reverseBits(self, n: int) -> int:
        ans = 0
        power = 31
        while n:
            ans += (n & 1) << power
            n >>= 1
            power -= 1
        return ans


    # 191. Number of 1 Bits
    def hammingWeight(self, n: int) -> int:
        ans = 0
        while n:
            ans += n & 1
            n >>= 1
        return ans


    # 198. House Robber
    def rob(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 0:
            return 0
        if n == 1:
            return nums[0]
        dp = [0] * (n + 1)
        dp[1] = nums[0]
        for i in range(2, n + 1):
            dp[i] = max(dp[i - 1], dp[i - 2] + nums[i - 1])
        return dp[n]


    # 199. Binary Tree Right Side View
    def rightSideView(self, root: TreeNode) -> List[int]:
        if not root:
            return []
        cur = [root]
        ans = []
        while cur:
            next = []
            for i in range(len(cur)):
                node = cur[i]
                if node.left:
                    next.append(node.left)
                if node.right:
                    next.append(node.right)
                if i == len(cur) - 1:
                    ans.append(node.val)
            cur = next
        return ans


    # 200. Number of Islands
    def numIslands(self, grid: List[List[str]]) -> int:
        if not grid:
            return 0
        count = 0
        dq = deque()
        dircs = [[0, 1], [0, -1], [1, 0], [-1, 0]]
        m, n = len(grid), len(grid[0])
        for i in range(m):
            for j in range(n):
                if grid[i][j] == "1":
                    dq.append([i, j])
                    # grid[i][j] = "0"
                    while dq:
                        x, y = dq.popleft()
                        for dirc in dircs:
                            p, q = x + dirc[0], y + dirc[1]
                            if (
                                p >= 0
                                and p < m
                                and q >= 0
                                and q < n
                                and grid[p][q] == "1"
                            ):
                                grid[p][q] = "0"
                                dq.append([p, q])
                    count += 1
        return count


    # 202. Happy Number
    def isHappy(self, n: int) -> bool:
        s = set()
        while True:
            n = sum([int(x) ** 2 for x in str(n)])
            if n == 1:
                return True
            if n in s:
                return False
            s.add(n)


    # 203. Remove Linked List Elements
    def removeElements(self, head: ListNode, val: int) -> ListNode:
        dummy = ListNode(float("-Inf"))
        dummy.next = head
        p = dummy
        while p.next:
            if p.next.val == val:
                p.next = p.next.next
            else:
                p = p.next
        return dummy.next


    # 204. Count Primes
    def countPrimes(self, n: int) -> int:
        if n < 2:
            return 0
        arr = [True] * n
        arr[0] = arr[1] = False
        i = 2
        while i * i < n:
            if arr[i]:
                j = i * i
                while j < n:
                    arr[j] = False
                    j += i
            i += 1
        return sum(arr)


    # 205. Isomorphic Strings
    def isIsomorphic(self, s: str, t: str) -> bool:
        if len(s) != len(t):
            return False
        map, mapped = {}, set()
        for char1, char2 in zip(s, t):
            if char1 in map:
                if char2 != map[char1]:
                    return False
            elif char2 in mapped:
                return False
            else:
                map[char1] = char2
                mapped.add(char2)
        return True


    # 206. Reverse Linked List
    def reverseList(self, head: ListNode) -> ListNode:
        prev, curr = None, head
        while curr:
            next = curr.next
            curr.next = prev
            prev = curr
            curr = next
        return prev


    # 207. Course Schedule
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        in_degrees = [0] * numCourses
        graph = defaultdict(set)
        for second_course, first_course in prerequisites:
            if first_course in graph[second_course]:
                return False
            if second_course not in graph[first_course]:
                graph[first_course].add(second_course)
                in_degrees[second_course] += 1
        q = deque([course for course in range(numCourses) if in_degrees[course] == 0])
        while q:
            course = q.popleft()
            if course in graph:
                for nbr in graph[course]:
                    in_degrees[nbr] -= 1
                    if in_degrees[nbr] == 0:
                        q.append(nbr)
        return not len(
            [course for course in range(numCourses) if in_degrees[course] > 0]
        )


    # 209. Minimum Size Subarray Sum
    def minSubArrayLen(self, s: int, nums: List[int]) -> int:
        i = cumsum = 0
        ans = float("inf")
        for j, num in enumerate(nums):
            cumsum += num
            while cumsum >= s:
                ans = min(ans, j - i + 1)
                cumsum -= nums[i]
                i += 1
        return ans if ans < float("inf") else 0


    # 210. Course Schedule II
    def findOrder(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:
        graph = collections.defaultdict(set)
        indegrees = [0] * numCourses
        for second, first in prerequisites:
            if first in graph[second]:
                return []
            if second not in graph[first]:
                graph[first].add(second)
                indegrees[second] += 1
        ans = []
        q = collections.deque(
            [course for course, degree in enumerate(indegrees) if degree == 0]
        )
        while q:
            course = q.popleft()
            ans.append(course)
            for next_course in graph[course]:
                indegrees[next_course] -= 1
                if indegrees[next_course] == 0:
                    q.append(next_course)
        return ans if len(ans) == numCourses else []


    # 212. Word Search II
    def findWords(self, board: List[List[str]], words: List[str]) -> List[str]:
        trie = {}
        for word in words:
            node = trie
            for char in word:
                if char not in node:
                    node[char] = {}
                node = node[char]
            node["$"] = word
        ans = set()
        m, n = len(board), len(board[0])
        directions = [[-1, 0], [1, 0], [0, -1], [0, 1]]
        def backtrack(i, j, node):
            if 0 <= i < m and 0 <= j < n and board[i][j] in node:
                ch = board[i][j]
                cur_node = node[ch]

                if "$" in cur_node:
                    ans.add(cur_node["$"])
                # if not cur_node:
                #    node.pop(ch)
                #    return

                board[i][j] = "*"
                for d in directions:
                    backtrack(i + d[0], j + d[1], cur_node)
                board[i][j] = ch
        for i in range(m):
            for j in range(n):
                backtrack(i, j, trie)
        return list(ans)


    # 215. Kth Largest Element in an Array
    def findKthLargest(self, nums: List[int], k: int) -> int:        
        def partition(l, r):
            idx = random.randint(l, r)
            nums[r], nums[idx] = nums[idx], nums[r]
            i = j = l
            while i < r:
                if nums[i] <= nums[r]:
                    nums[i], nums[j] = nums[j], nums[i]
                    j += 1
                i += 1
            nums[j], nums[r] = nums[r], nums[j]
            return j
        def sort(l, r, k):
            if l == r:
                return
            pivot = partition(l, r)
            if pivot < k:
                sort(pivot + 1, r, k)
            elif pivot > k:
                sort(l, pivot - 1, k)            
        n = len(nums)
        sort(0, n - 1, n - k)
        return nums[n - k]


    # 217. Contains Duplicate
    def containsDuplicate(self, nums: List[int]) -> bool:
        return len(set(nums)) != len(nums)


    # 218. The Skyline Problem
    def getSkyline(self, buildings: List[List[int]]) -> List[List[int]]:
        ans = [[-1, 0]]
        position = set([b[0] for b in buildings] + [b[1] for b in buildings])
        live = []
        i = 0
        n = len(buildings)
        for t in sorted(position):
            # add the new buildings whose left side is lefter than position t
            while i < n and buildings[i][0] <= t:
                heapq.heappush(live, (-buildings[i][2], buildings[i][1]))
                i += 1
            # remove the past buildings whose right side is lefter than position t
            while live and live[0][1] <= t:
                heapq.heappop(live)
            # pick the highest existing building at this moment
            h = -live[0][0] if live else 0
            if ans[-1][1] != h:
                ans.append([t, h])
        return ans[1:]


    # 219. Contains Duplicate II
    def containsNearbyDuplicate(self, nums: List[int], k: int) -> bool:
        dict = {}
        for i, num in enumerate(nums):
            if num in dict and i - dict[num] <= k:
                return True
            dict[num] = i
        return False


    # 221. Maximal Square
    def maximalSquare(self, matrix: List[List[str]]) -> int:
        if not matrix:
            return 0
        m, n = len(matrix), len(matrix[0])
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        length = 0
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                dp[i][j] = (
                    min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1
                    if matrix[i - 1][j - 1] == "1"
                    else 0
                )
                length = max(length, dp[i][j])
        return length * length


    # 222. Count Complete Tree Nodes
    def countNodes(self, root: TreeNode) -> int:
        def getHeight(root):
            if not root:
                return 0
            return 1 + getHeight(root.left)
        if not root:
            return 0
        height = getHeight(root)
        rheight = getHeight(root.right)
        if rheight == height - 1:
            return (1 << (height - 1)) + self.countNodes(root.right)
        else:
            return (1 << rheight) + self.countNodes(root.left)


    # 224. Basic Calculator
    def calculate(self, s: str) -> int:
        stack = []
        ans = num = 0
        sign = 1
        for char in s:
            if char.isdigit():
                num = 10 * num + int(char)
            elif char == "+":
                ans += num * sign
                num = 0
                sign = 1
            elif char == "-":
                ans += num * sign
                num = 0
                sign = -1
            elif char == "(":
                stack.append(ans)
                stack.append(sign)
                ans = 0
                sign = 1
            elif char == ")":
                ans += sign * num
                ans *= stack.pop()
                ans += stack.pop()
                num = 0
                sign = 1
        return ans + num * sign


    # 226. Invert Binary Tree
    def invertTree(self, root):
        """
        :type root: TreeNode
        :rtype: TreeNode
        """
        if root is not None:
            root.left, root.right = self.invertTree(root.right), self.invertTree(
                root.left
            )
        return root


    # 227. Basic Calculator II
    def calculate2(self, s: str) -> int:
        stack = []
        s += "$"
        num = 0
        sign = "+"
        for char in s:
            if char == " ":
                continue
            elif char.isdigit():
                num = 10 * num + int(char)
            else:
                if sign == "+":
                    stack.append(num if sign == "+" else -num)
                elif sign == "-":
                    stack.append(num if sign == "+" else -num)
                elif sign == "*":
                    stack.append(stack.pop() * num)
                elif sign == "/":
                    stack.append(int(stack.pop() / num))
                num = 0
                sign = char
        print(stack)
        return sum(stack)


    # 228. Summary Ranges
    def summaryRanges(self, nums: List[int]) -> List[str]:
        if not nums:
            return []
        nums.append(float("inf"))
        lo = hi = nums[0]
        ans = []
        for i in range(1, len(nums)):
            if nums[i] > hi + 1:
                if lo == hi:
                    ans.append(str(lo))
                else:
                    ans.append(f"{lo}->{hi}")
                lo = hi = nums[i]
            else:
                hi = nums[i]
        return ans


    # 229. Majority Element II
    def majorityElement2(self, nums: List[int]) -> List[int]:
        cand1 = cand2 = None
        count1 = count2 = 0
        for num in nums:
            if num == cand1:
                count1 += 1
            elif num == cand2:
                count2 += 1
            elif count1 == 0:
                cand1 = num
                count1 = 1
            elif count2 == 0:
                cand2 = num
                count2 = 1
            else:
                count1 -= 1
                count2 -= 1
        ans = []
        n = len(nums)
        if nums.count(cand1) > n // 3:
            ans.append(cand1)
        if nums.count(cand2) > n // 3:
            ans.append(cand2)
        return ans


    # 230. Kth Smallest Element in a BST
    def kthSmallest(self, root: TreeNode, k: int) -> int:
        ans = []
        def inorder(root):
            if not root:
                return
            inorder(root.left)
            if len(ans) == k:
                return
            ans.append(root.val)
            inorder(root.right)
        inorder(root)
        return ans[-1]


    # 231. Power of Two
    def isPowerOfTwo(self, n: int) -> bool:
        return n and n & (n - 1) == 0


    # 234. Palindrome Linked List
    def isPalindrome(self, head: ListNode) -> bool:
        def findEndOfHalf(node):
            slow = fast = node
            while fast.next and fast.next.next:
                slow = slow.next
                fast = fast.next.next
            return slow
        def reverseList(node):
            prev, curr = None, node
            while curr:
                next = curr.next
                curr.next = prev
                prev = curr
                curr = next
            return prev
        def compareTwoLists(node1, node2):
            while node1 and node2:
                if node1.val != node2.val:
                    return False
                node1 = node1.next
                node2 = node2.next
            return True
        if not head:
            return True
        end_of_first_half = findEndOfHalf(head)
        end_of_second_half = reverseList(end_of_first_half.next)
        return compareTwoLists(head, end_of_second_half)


    # 235. Lowest Common Ancestor of a Binary Search Tree
    def lowestCommonAncestor(
        self, root: "TreeNode", p: "TreeNode", q: "TreeNode"
    ) -> "TreeNode":
        if not root:
            return
        if p.val < root.val and q.val < root.val:
            return self.lowestCommonAncestor(root.left, p, q)
        if p.val > root.val and q.val > root.val:
            return self.lowestCommonAncestor(root.right, p, q)
        return root


    # 236. Lowest Common Ancestor of a Binary Tree
    def lowestCommonAncestor(
        self, root: "TreeNode", p: "TreeNode", q: "TreeNode"
    ) -> "TreeNode":
        if not root:
            return
        if root.val == p.val or root.val == q.val:
            return root
        left = self.lowestCommonAncestor(root.left, p, q)
        right = self.lowestCommonAncestor(root.right, p, q)
        if left and not right:
            return left
        if not left and right:
            return right
        if left and right:
            return root


    # 237. Delete Node in a Linked List
    def deleteNode(self, node):
        """
        :type node: ListNode
        :rtype: void Do not return anything, modify node in-place instead.
        """
        node.val = node.next.val
        node.next = node.next.next


    # 238. Product of Array Except Self
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        res1, res2 = [1], [1]
        n = len(nums)
        cum_prod = 1
        for i in range(n):
            cum_prod *= nums[i]
            res1.append(cum_prod)
        cum_prod = 1
        for i in range(n - 1, -1, -1):
            cum_prod *= nums[i]
            res2.append(cum_prod)
        res = []
        for i in range(n):
            res.append(res1[i] * res2[n - i - 1])
        return res


    # 239. Sliding Window Maximum
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        if not nums:
            return []
        q = collections.deque()
        ans = []
        for i, num in enumerate(nums):
            if q and q[0] <= i - k:
                q.popleft()
            while q and nums[q[-1]] <= num:
                q.pop()
            q.append(i)
            if i >= k - 1:
                ans.append(nums[q[0]])
        return ans


    # 240. Search a 2D Matrix II
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        if not matrix:
            return False
        m, n = len(matrix), len(matrix[0])
        row = 0
        col = n - 1
        while col >= 0 and row < m:
            if matrix[row][col] > target:
                col -= 1
            elif matrix[row][col] < target:
                row += 1
            else:
                return True
        return False


    # 242. Valid Anagram
    def isAnagram(self, s: str, t: str) -> bool:
        return Counter(s) == Counter(t)


    # 257. Binary Tree Paths
    def binaryTreePaths(self, root: TreeNode) -> List[str]:
        def helper(root, path):
            if not root:
                return
            if not root.left and not root.right:
                path += str(root.val)
                ans.append(path)
            else:
                path += str(root.val) + "->"
                helper(root.left, path)
                helper(root.right, path)
        ans = []
        helper(root, "")
        return ans


    # 258. Add Digits
    def addDigits(self, num: int) -> int:
        num = str(num)
        while len(num) > 1:
            res = 0
            for d in num:
                res += int(d)
            num = str(res)
        return int(num)


    # 268. Missing Number
    def missingNumber(self, nums: List[int]) -> int:
        ans = len(nums)
        for i, num in enumerate(nums):
            ans += i - num
        return ans


    # 273. Integer to English Words
    def numberToWords(self, num: int) -> str:
        to19 = [
            "One",
            "Two",
            "Three",
            "Four",
            "Five",
            "Six",
            "Seven",
            "Eight",
            "Nine",
            "Ten",
            "Eleven",
            "Twelve",
            "Thirteen",
            "Fourteen",
            "Fifteen",
            "Sixteen",
            "Seventeen",
            "Eighteen",
            "Nineteen",
        ]
        tens = [
            "Twenty",
            "Thirty",
            "Forty",
            "Fifty",
            "Sixty",
            "Seventy",
            "Eighty",
            "Ninety",
        ]
        def words(n):
            if n == 0:
                return []
            if n < 20:
                return [to19[n - 1]]
            if n < 100:
                return [tens[n // 10 - 2]] + words(n % 10)
            if n < 1000:
                return [to19[n // 100 - 1]] + ["Hundred"] + words(n % 100)
            for p, w in enumerate(("Thousand", "Million", "Billion"), 1):
                if n < 1000 ** (p + 1):
                    return words(n // 1000 ** p) + [w] + words(n % 1000 ** p)
        return " ".join(words(num)) or "Zero"


    # 278. First Bad Version
    def firstBadVersion(self, n: int) -> int:
        # find the leftmost boundary of bad
        l, r = 1, n + 1
        while l < r:
            mid = l + (r - l) // 2
            if not isBadVersion(mid):
                l = mid + 1
            else:
                r = mid
        return l


    # 279. Perfect Squares
    def numSquares(self, n: int) -> int:
        squares = [x ** 2 for x in range(1, int(n ** 0.5) + 1)]
        dp = list(range(n + 1))
        for i in range(1, n + 1):
            for square in squares:
                if i >= square:
                    dp[i] = min(dp[i], dp[i - square] + 1)
        return dp[n]


    # 282. Expression Add Operators
    def addOperators(self, num: str, target: int) -> List[str]:
        n = len(num)
        ans = []
        def backtrack(idx, prev_num, prev_res, path):
            if idx == n:
                if prev_res == target:
                    ans.append(path)
                return
            for i in range(idx, n):
                num_str = num[idx : i + 1]
                if len(num_str) > 1 and num_str[0] == "0":
                    continue
                num_int = int(num_str)
                if idx == 0:
                    backtrack(i + 1, num_int, num_int, num_str)
                else:
                    backtrack(i + 1, num_int, prev_res + num_int, path + "+" + num_str)
                    backtrack(i + 1, -num_int, prev_res - num_int, path + "-" + num_str)
                    backtrack(
                        i + 1,
                        prev_num * num_int,
                        prev_res - prev_num + prev_num * num_int,
                        path + "*" + num_str,
                    )
        backtrack(0, 0, 0, "")
        return ans


    # 283. Move Zeroes
    def moveZeroes(self, nums: List[int]) -> None:
        j = 0
        for i in range(len(nums)):
            if nums[i] != 0:
                nums[i], nums[j] = nums[j], nums[i]
                j += 1


    # 287. Find the Duplicate Number
    def findDuplicate(self, nums: List[int]) -> int:
        for i in range(len(nums)):
            index = abs(nums[i]) - 1
            if nums[index] > 0:
                nums[index] *= -1
            else:
                return index + 1


    # 289. Game of Life
    def gameOfLife(self, boar: List[List[int]]) -> None:
        m = len(board)
        n = len(board[0]) if m else 0
        for i in xrange(m):
            for j in xrange(n):
                count = 0
                for I in xrange(max(i - 1, 0), min(i + 2, m)):
                    for J in xrange(max(j - 1, 0), min(j + 2, n)):
                        count += board[I][J] % 2
                if board[i][j]:
                    if count < 3 or count > 4:
                        board[i][j] = 3  # live -> dead
                else:
                    if count == 3:
                        board[i][j] = 2  # dead -> live
        for i in xrange(m):
            for j in xrange(n):
                if board[i][j] == 2:
                    board[i][j] = 1
                if board[i][j] == 3:
                    board[i][j] = 0


    # 290. Word Pattern
    def wordPattern(self, pattern: str, str: str) -> bool:
        word = str.split()
        if len(word) != len(pattern):
            return False
        map_pattern, map_word = {}, {}
        for i in xrange(len(word)):
            if pattern[i] not in map_pattern:
                map_pattern[pattern[i]] = word[i]
            if word[i] not in map_word:
                map_word[word[i]] = pattern[i]
            if map_word[word[i]] != pattern[i] or map_pattern[pattern[i]] != word[i]:
                return False
        return True


    # 292. Nim Game
    def canWinNim(self, n: int) -> bool:
        return n % 4 != 0


    # 299. Bulls and Cows
    def getHint(self, secret: str, guess: str) -> str:
        bull, cow = 0, 0
        count = {}
        for i in xrange(len(secret)):
            if secret[i] == guess[i]:
                bull += 1
            if secret[i] in count:
                count[secret[i]] += 1
            else:
                count[secret[i]] = 1
        for i in xrange(len(guess)):
            if guess[i] in count:
                if count[guess[i]] > 0:
                    cow += 1
                    count[guess[i]] -= 1
        cow = cow - bull
        return str(bull) + "A" + str(cow) + "B"


    # 300. Longest Increasing Subsequence
    def lengthOfLIS(self, nums: List[int]) -> int:
        if not nums:
            return 0
        n = len(nums)
        dp = [1] * n
        for i in range(1, n):
            for j in range(i):
                if nums[j] < nums[i]:
                    dp[i] = max(dp[i], dp[j] + 1)
        return max(dp)


    # 301. Remove Invalid Parentheses
    def removeInvalidParentheses(self, s: str) -> List[str]:
        left = right = 0
        for ch in s:
            if ch == "(":
                left += 1
            elif ch == ")":
                if left > 0:
                    left -= 1
                else:
                    right += 1
        n = len(s)
        ans = set()
        def dfs(idx, left_added, right_added, left_to_remove, right_to_remove, path):
            if idx == n:
                if left_to_remove == right_to_remove == 0 and left_added == right_added:
                    ans.add(path)
                return
            if s[idx] == "(":
                if left_to_remove > 0:
                    dfs(
                        idx + 1,
                        left_added,
                        right_added,
                        left_to_remove - 1,
                        right_to_remove,
                        path,
                    )
                dfs(
                    idx + 1,
                    left_added + 1,
                    right_added,
                    left_to_remove,
                    right_to_remove,
                    path + "(",
                )
            elif s[idx] == ")":
                if left_added > right_added:
                    dfs(
                        idx + 1,
                        left_added,
                        right_added + 1,
                        left_to_remove,
                        right_to_remove,
                        path + ")",
                    )
                if right_to_remove > 0:
                    dfs(
                        idx + 1,
                        left_added,
                        right_added,
                        left_to_remove,
                        right_to_remove - 1,
                        path,
                    )
            else:
                dfs(
                    idx + 1,
                    left_added,
                    right_added,
                    left_to_remove,
                    right_to_remove,
                    path + s[idx],
                )

        dfs(0, 0, 0, left, right, "")
        return list(ans) if ans else [""]


    # 309. Best Time to Buy and Sell Stock with Cooldown
    def maxProfit(self, prices: List[int]) -> int:
        sold, held, reset = float("-inf"), float("-inf"), 0
        for price in prices:
            pre_sold = sold
            sold = held + price
            held = max(held, reset - price)
            reset = max(reset, pre_sold)
        return max(reset, sold)


    # 310. Minimum Height Trees
    def findMinHeightTrees(self, n: int, edges: List[List[int]]) -> List[int]:
        if n <= 2:
            return list(range(n))
        graph = [set() for _ in range(n)]
        for s, e in edges:
            graph[s].add(e)
            graph[e].add(s)
        leaves = [i for i in range(n) if len(graph[i]) == 1]
        remaining_leaves = n
        while remaining_leaves > 2:
            remaining_leaves -= len(leaves)
            new_leaves = []
            while leaves:
                leaf = leaves.pop()
                nbr = graph[leaf].pop()
                graph[nbr].remove(leaf)
                if len(graph[nbr]) == 1:
                    new_leaves.append(nbr)
            leaves = new_leaves
        return leaves


    # 315. Count of Smaller Numbers After Self
    def countSmaller(self, nums: List[int]) -> List[int]:
        def rightrotate(root):
            a = root.left
            b = root.left.right
            a.right = root
            root.left = b
            return a
        def leftrotate(root):
            a = root.right
            b = root.right.left
            a.left = root
            root.right = b
            return a
        def insert(v, root):
            if not root:
                return AVL(v)
            if v <= root.val:
                root.left = insert(v, root.left)
            else:
                self.count += root.left.size + 1 if root.left else 1
                root.right = insert(v, root.right)
            height = 0
            size = 0
            indi = 0
            if root.left:
                height = root.left.height
                size += root.left.size
                indi = root.left.height
            if root.right:
                height = max(height, root.right.height)
                size = size + root.right.size
                indi -= root.right.height
            height += 1
            size += 1
            root.size = size
            root.height = height
            if indi <= 1 or indi >= -1:
                return root
            if indi > 1 and root.left and v <= root.left.val:
                return rightrotate(root)
            if indi > 1 and root.left and v > root.left.val:
                root.left = leftrotate(root.left)
                return rightrotate(root)
            if indi < -1 and root.right and v > root.right.val:
                return leftrotate(root)
            if indi < -1 and root.right and v <= root.right.val:
                root.right = rightrotate(root.right)
                return leftrotate(root)
        def printr(root):
            if not root:
                return
            printr(root.left)
            print(root.val)
            printr(root.right)
        ans = [0] * len(nums)
        root = None
        for i in range(len(nums) - 1, -1, -1):
            self.count = 0
            root = insert(nums[i], root)
            ans[i] = self.count
        # printr(root)
        return ans


    # 319. Bulb Switcher
    def bulbSwitch(self, n: int) -> int:
        return int(math.sqrt(n))


    # 322. Coin Change
    def coinChange(self, coins: List[int], amount: int) -> int:
        memo = {0: 0}
        def dp(n):
            if n in memo:
                return memo[n]
            ans = float("inf")
            for coin in coins:
                if n - coin >= 0:
                    ans = min(ans, 1 + dp(n - coin))
            memo[n] = ans
            return memo[n]
        ans = dp(amount)
        return ans if ans < float("inf") else -1


    # 326. Power of Three
    def isPowerOfThree(self, n: int) -> bool:
        return n > 0 and 1162261467 % n == 0


    # 328. Odd Even Linked List
    def oddEvenList(self, head: ListNode) -> ListNode:
        if not head:
            return head
        node1 = head
        node2 = head.next
        dummy = node2
        while node1 and node2:
            node1.next = node2.next
            node1 = node1.next
            if node1:
                node2.next = node1.next
                node2 = node2.next
        node1 = head
        while node1.next:
            node1 = node1.next
        node1.next = dummy
        return head


    # 329. Longest Increasing Path in a Matrix
    def longestIncreasingPath(self, matrix: List[List[int]]) -> int:
        if not matrix:
            return 0
        m, n = len(matrix), len(matrix[0])
        memo = [[0] * n for _ in range(m)]
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        def dfs(i, j):
            if memo[i][j] > 0:
                return memo[i][j]
            for d in directions:
                next_i, next_j = i + d[0], j + d[1]
                if (
                    0 <= next_i < m
                    and 0 <= next_j < n
                    and matrix[next_i][next_j] > matrix[i][j]
                ):
                    memo[i][j] = max(memo[i][j], dfs(next_i, next_j))
            memo[i][j] += 1
            return memo[i][j]
        ans = 0
        for i in range(m):
            for j in range(n):
                ans = max(ans, dfs(i, j))
        return ans


    # 332. Reconstruct Itinerary
    def findItinerary(self, tickets: List[List[str]]) -> List[str]:

        """
        targets = collections.defaultdict(list)

        for a, b in reversed(sorted(tickets)):
            targets[a].append(b)
        ans = []
        def visit(airport):
            while targets[airport]:
                visit(targets[airport].pop())
            ans.append(airport)
        visit('JFK')
        return ans[::-1]
        """
        graph = collections.defaultdict(list)
        for origin, dest in tickets:
            graph[origin].append(dest)
        visited = {}
        for origin, itinerary in graph.items():
            itinerary.sort()
            visited[origin] = [False] * len(itinerary)
        n = len(tickets)
        self.ans = []
        def backtracking(origin, route):
            if len(route) == n + 1:
                self.ans = route
                return True
            for i, next in enumerate(graph[origin]):
                if not visited[origin][i]:
                    visited[origin][i] = True
                    ret = backtracking(next, route + [next])
                    visited[origin][i] = False
                    if ret:
                        return True
            return False
        backtracking("JFK", ["JFK"])
        return self.ans


    # 334. Increasing Triplet Subsequence
    def increasingTriplet(self, nums: List[int]) -> bool:
        first_min = second_min = float("inf")
        for num in nums:
            if num <= first_min:
                first_min = num
            elif num <= second_min:
                second_min = num
            else:
                return True
        return False


    # 336. Palindrome Pairs
    def palindromePairs(self, words: List[str]) -> List[List[int]]:
        def isPalindrome(word):
            return word == word[::-1]
        ans = []
        words = {word: i for i, word in enumerate(words)}
        for word, i in words.items():
            n = len(word)
            for j in range(n + 1):
                prefix, suffix = word[:j], word[j:]
                if isPalindrome(prefix):
                    suffix = suffix[::-1]
                    if suffix != word and suffix in words:
                        ans.append([words[suffix], i])
                if j != n and isPalindrome(suffix):  # j != n remove duplicates
                    prefix = prefix[::-1]
                    if prefix != word and prefix in words:
                        ans.append([i, words[prefix]])
        return ans


    # 338. Counting Bits
    def countBits(self, num: int) -> List[int]:
        res = [0] * (num + 1)
        for i in range(1, num + 1):
            res[i] = res[i & (i - 1)] + 1
        return res


    # 342. Power of Four
    def isPowerOfFour(self, num: int) -> bool:
        if num <= 0:
            return False
        while (num % 4 ==0):
            num /= 4
        return num == 1


    # 344. Reverse String
    def reverseString(self, s: List[str]) -> None:
        i, j = 0, len(s) - 1
        while i < j:
            s[i], s[j] = s[j], s[i]
            i += 1
            j -= 1


    # 345. Reverse Vowels of a String
    def reverseVowels(self, s: str) -> str:
        l = list(s)
        i, j = 0, len(l) - 1
        while i < j:
            while i < j and l[i] not in 'aeiouAEIOU':
                i += 1
            while i < j and l[j] not in 'aeiouAEIOU':
                j -= 1
            if i < j:
                l[i], l[j] = l[j], l[i]
                i += 1
                j -= 1
        return "".join(l)


    # 347. Top K Frequent Elements
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        counter = collections.Counter(nums)
        nums = list(counter.keys())
        def partition(l, r):
            idx = random.randint(l, r)
            nums[idx], nums[r] = nums[r], nums[idx]
            i = l
            for j in range(i, r):
                if counter[nums[j]] < counter[nums[r]]:
                    nums[i], nums[j] = nums[j], nums[i]
                    i += 1
            nums[i], nums[r] = nums[r], nums[i]
            return i
        def sort(l, r, k):
            if l == r:
                return
            pivot = partition(l, r)
            if pivot == k:
                return
            elif pivot < k:
                sort(pivot + 1, r, k)
            else:
                sort(l, pivot - 1, k)
        n = len(nums)
        sort(0, n - 1, n - k)
        return nums[n - k :]


    # 349. Intersection of Two Arrays
    def intersection(self, nums1: List[int], nums2: List[int]) -> List[int]:
        s1 = set(nums1)
        s2 = set(nums2)
        return [x for x in s1 if x in s2]


    # 350. Intersection of Two Arrays II
    def intersect(self, nums1: List[int], nums2: List[int]) -> List[int]:
        c1 = Counter(nums1)
        c2 = Counter(nums2)
        res = []
        for x in c1:
            if x in c2:
                res.extend([x] * min(c1[x], c2[x]))
        return res


    # 367. Valid Perfect Square
    def isPerfectSquare(self, num: int) -> bool:
        left, right = 1, num
        while left <= right:
            mid = (left + right) / 2
            if mid >= num / mid:
                right = mid - 1
            else:
                left = mid + 1
        return left == num / left and num % left == 0


    # 371. Sum of Two Integers
    def getSum(self, a: int, b: int) -> int:
        mask = 0xFFFFFFFF
        while b:
            carry = (a & b) << 1
            a = (a ^ b) & mask
            b = carry & mask
        max_int = 0x7FFFFFFF
        return a if a < max_int else ~(a ^ mask)


    # 374. Guess Number Higher or Lower
    def guessNumber(self, n: int) -> int:
        left, right = 1, n
        while True:
            mid = (left + right) / 2
            trial = guess(mid)
            if trial == -1:
                right = mid - 1
            elif trial == 1:
                left = mid + 1
            else:
                return mid


    # 378. Kth Smallest Element in a Sorted Matrix
    def kthSmallest(self, matrix: List[List[int]], k: int) -> int:
        m, n = len(matrix), len(matrix[0])
        h = []
        for i in range(min(k, m)):
            h.append([matrix[i][0], i, 0])
        heapq.heapify(h)
        while k:
            val, i, j = heapq.heappop(h)
            if j < n - 1:
                heapq.heappush(h, [matrix[i][j + 1], i, j + 1])
            k -= 1
        return val

    
    # 382. Linked List Random Node
    def getRandom(self) -> int:
        node = self.head
        count = 0
        val = None
        while node:
            p = random.randint(0, count)
            if p == 0:
                val = node.val
            node = node.next
            count += 1
        return val


    # 383. Ransom Note
    def canConstruct(self, ransomNote: str, magazine: str) -> bool:
        countR = collections.Counter(ransomNote)
        countM = collections.Counter(magazine)
        return not countR - countM
        

    # 387. First Unique Character in a String
    def firstUniqChar(self, s: str) -> int:
        cnt = Counter(s)
        for i, char in enumerate(s):
            if cnt[char] == 1:
                return i
        return -1


    # 389. Find the Difference
    def findTheDifference(self, s: str, t: str) -> str:
        return (collections.Counter(t) - collections.Counter(s)).keys()[0]


    # 392. Is Subsequence
    def isSubsequence(self, s: str, t: str) -> bool:
        if not s:
            return True
        i = 0
        for j in range(len(t)):
            if t[j] == s[i]:
                i += 1
                if i == len(s):
                    return True
        return False


    # 394. Decode String
    def decodeString(self, s: str) -> str:
        num = 0
        string = ""
        num_stack = []
        string_stack = []
        for char in s:
            if char.isdigit():
                num = 10 * num + int(char)
            elif char.isalpha():
                string += char
            elif char == "[":
                num_stack.append(num)
                string_stack.append(string)
                num = 0
                string = ""
            elif char == "]":
                string = string_stack.pop() + string * num_stack.pop()
        return string


    # 395. Longest Substring with At Least K Repeating Characters
    def longestSubstring(self, s: str, k: int) -> int:
        if len(s) < k:
            return 0
        ans = 0
        counter = collections.Counter(s)
        for char, count in counter.items():
            if count < k:
                for sub in s.split(char):
                    ans = max(ans, self.longestSubstring(sub, k))
                return ans
        return len(s)


    # 399. Evaluate Division
    def calcEquation(
        self, equations: List[List[str]], values: List[float], queries: List[List[str]]
    ) -> List[float]:
        # build graph
        graph = defaultdict(list)
        shortcut = {}
        for (a, b), value in zip(equations, values):
            graph[a].append([b, value])
            graph[b].append([a, 1 / value])

        # find path
        def find_path(query):
            a, b = query
            if a not in graph or b not in graph:
                return -1
            q = deque([(a, 1)])
            visited = set()
            visited.add(a)
            while q:
                front, curr_prod = q.popleft()
                if front == b:
                    return curr_prod
                for nbr, val in graph[front]:
                    if nbr not in visited:
                        q.append((nbr, curr_prod * val))
                        visited.add(front)
            return -1

        return [find_path(query) for query in queries]


    # 400. Nth Digit
    def findNthDigit(self, n: int) -> int:
        count = 9
        start = 1
        length = 1
        while n > length * count:
            n -= length * count
            length += 1
            count *= 10
            start *= 10
        start += (n - 1) // length
        return str(start)[(n - 1) % length]


    # 404. Sum of Left Leaves
    def sumOfLeftLeaves(self, root: TreeNode) -> int:
        self.ans = 0
        def helper(root):
            if root:
                if root.left and not root.left.left and not root.left.right:
                    self.ans += root.left.val
                helper(root.left)
                helper(root.right)
        helper(root)
        return self.ans


    # 406. Queue Reconstruction by Height
    def reconstructQueue(self, people: List[List[int]]) -> List[List[int]]:
        people.sort(key=lambda x: (-x[0], x[1]))
        ans = []
        for h, k in people:
            ans.insert(k, [h, k])
        return ans


    # 410. Split Array Largest Sum
    def splitArray(self, nums: List[int], m: int) -> int:
        l = 0
        h = 0
        for num in nums:
            l = max(num, l)
            h = h + num

        while l < h:
            mid = (l + h) // 2
            count = 0
            running_sum = 0
            for num in nums:
                running_sum = running_sum + num
                if running_sum > mid:
                    running_sum = num
                    count += 1
            if count + 1 > m:
                l = mid + 1
            else:
                h = mid
        return l


    # 412. Fizz Buzz
    def fizzBuzz(self, n: int) -> List[str]:
        ans = []
        for i in range(1, n + 1):
            filter_1 = i % 3 == 0
            filter_2 = i % 5 == 0
            if filter_1 and not filter_2:
                ans.append("Fizz")
            elif not filter_1 and filter_2:
                ans.append("Buzz")
            elif filter_1 and filter_2:
                ans.append("FizzBuzz")
            else:
                ans.append(str(i))
        return ans


    # 414. Third Maximum Number
    def thirdMax(self, nums: List[int]) -> int:
        first, second, third = nums[0], float("-Inf"), float("-Inf")
        for i in xrange(1, len(nums)):
            if nums[i] > first:
                first, second, third = nums[i], first, second
            elif nums[i] < first and nums[i] > second:
                second, third = nums[i], second
            elif nums[i] < second and nums[i] > third:
                third = nums[i]
        return first if third == float("-Inf") else third


    # 415. Add Strings
    def addStrings(self, num1: str, num2: str) -> str:
        n1, n2 = len(num1), len(num2)
        ans = ""
        carry = 0
        while n1 > 0 and n2 > 0:
            tmp = int(num1[n1 - 1]) + int(num2[n2 - 1]) + carry
            ans += str(tmp % 10)
            carry = tmp // 10
            n1 -= 1
            n2 -= 1
        while n1 > 0:
            tmp = int(num1[n1 - 1]) + carry
            ans += str(tmp % 10)
            carry = tmp // 10
            n1 -= 1
        while n2 > 0:
            tmp = int(num2[n2 - 1]) + carry
            ans += str(tmp % 10)
            carry = tmp // 10
            n2 -= 1
        if carry > 0:
            ans += "1"
        return ans[::-1]


    # 416. Partition Equal Subset Sum
    def canPartition(self, nums: List[int]) -> bool:
        total_sum = sum(nums)
        if total_sum % 2 == 1:
            return False
        half_sum = total_sum // 2
        n = len(nums)
        dp = [[False] * (half_sum + 1) for _ in range(n + 1)]
        dp[0][0] = True
        for i in range(1, n + 1):
            for j in range(half_sum + 1):
                if j < nums[i - 1]:
                    dp[i][j] = dp[i - 1][j]
                else:
                    dp[i][j] = dp[i - 1][j] or dp[i - 1][j - nums[i - 1]]
        return dp[n][half_sum]


    # 419. Battleships in a Board
    def countBattleships(self, board: List[List[str]]) -> int:
        ans = 0
        m, n = len(board), len(board[0])
        for i in range(m):
            for j in range(n):
                if (
                    board[i][j] == "X"
                    and (i == 0 or board[i - 1][j] != "X")
                    and (j == 0 or board[i][j - 1] != "X")
                ):
                    ans += 1
        return ans


    # 430. Flatten a Multilevel Doubly Linked List
    def flatten(self, head: "Node") -> "Node":
        if not head:
            return head
        dummy = Node(0, None, head, None)
        stack = [head]
        prev = dummy
        while stack:
            curr = stack.pop()
            prev.next = curr
            curr.prev = prev
            if curr.next:
                stack.append(curr.next)
            if curr.child:
                stack.append(curr.child)
                curr.child = None  # don't forget
            prev = curr
        dummy.next.prev = None
        return dummy.next


    # 435. Non-overlapping Intervals
    def eraseOverlapIntervals(self, intervals: List[List[int]]) -> int:
        if not intervals:
            return 0
        intervals.sort()
        ans = 0
        start, end = intervals[0][0], intervals[0][1]
        for i in range(1, len(intervals)):
            if intervals[i][0] < end:
                ans += 1
                end = min(end, intervals[i][1])
            else:
                start, end = intervals[i][0], intervals[i][1]
        return ans


    # 438. Find All Anagrams in a String
    def findAnagrams(self, s: str, p: str) -> List[int]:
        n = len(p)
        counter_p = collections.Counter(p)
        counter_s = collections.Counter(s[: n - 1])
        ans = []
        for i in range(n - 1, len(s)):
            counter_s[s[i]] += 1
            if counter_s == counter_p:
                ans.append(i - n + 1)
            counter_s[s[i - n + 1]] -= 1
            if counter_s[s[i - n + 1]] == 0:
                del counter_s[s[i - n + 1]]
        return ans


    # 441. Arranging Coins
    def arrangeCoins(self, n: int) -> int:
        return int((-1 + math.sqrt(1 + 8 * n)) / 2)


    # 443. String Compression
    def compress(self, chars: List[str]) -> int:
        anchor = j = 0
        n = len(chars)
        for i in range(n):
            if i == n - 1 or chars[i + 1] != chars[i]:
                chars[j] = chars[anchor]
                j += 1
                if i > anchor:
                    for d in str(i - anchor + 1):
                        chars[j] = d
                        j += 1
                anchor = i + 1
        return j


    # 445. Add Two Numbers II
    def addTwoNumbers2(self, l1: ListNode, l2: ListNode) -> ListNode:
        def reverse(head):
            prev, curr = None, head
            while curr:
                next = curr.next
                curr.next = prev
                prev = curr
                curr = next
            return prev
        l1 = reverse(l1)
        l2 = reverse(l2)
        prev = None
        head = l1  # since nonempty
        carry = 0
        while l1 and l2:
            l1.val = l1.val + l2.val + carry
            carry = l1.val // 10
            l1.val = l1.val % 10
            prev = l1
            l1 = l1.next
            l2 = l2.next
        while l1:
            l1.val = l1.val + carry
            carry = l1.val // 10
            l1.val = l1.val % 10
            prev = l1
            l1 = l1.next
        if l2:
            prev.next = l2
            while l2:
                l2.val = l2.val + carry
                carry = l2.val // 10
                l2.val = l2.val % 10
                prev = l2
                l2 = l2.next
        if carry:
            prev.next = ListNode(1)
        return reverse(head)


    # 448. Find All Numbers Disappeared in an Array
    def findDisappearedNumbers(self, nums: List[int]) -> List[int]:
        for i in range(len(nums)):
            index = abs(nums[i]) - 1
            if nums[index] > 0:
                nums[index] *= -1
        ans = []
        for i, num in enumerate(nums):
            if nums[i] > 0:
                ans.append(i + 1)
        return ans


    # 451. Sort Characters By Frequency
    def frequencySort(self, s: str) -> str:
        counter = collections.Counter(s).most_common()
        return "".join([ch * val for ch, val in counter])


    # 452. Minimum Number of Arrows to Burst Balloons
    def findMinArrowShots(self, points: List[List[int]]) -> int:
        points.sort()
        ans = 0
        for i in range(len(points)):
            if i > 0 and points[i][0] <= end:
                end = min(end, points[i][1])
            else:
                ans += 1
                end = points[i][1]
        return ans


    # 453. Minimum Moves to Equal Array Elements
    def minMoves(self, nums: List[int]) -> int:
        return sum(nums) - min(nums) * len(nums)


    # 461. Hamming Distance
    def hammingDistance(self, x: int, y: int) -> int:
        xor = x ^ y
        ans = 0
        while xor:
            ans += xor & 1
            xor >>= 1
        return ans


    # 463. Island Perimeter
    def islandPerimeter(self, grid: List[List[int]]) -> int:
        ans = 0
        m, n = len(grid), len(grid[0])
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 1:
                    if i == 0 or grid[i - 1][j] == 0:
                        ans += 1
                    if i == m - 1 or grid[i + 1][j] == 0:
                        ans += 1
                    if j == 0 or grid[i][j - 1] == 0:
                        ans += 1
                    if j == n - 1 or grid[i][j + 1] == 0:
                        ans += 1
        return ans


    # 468. Validate IP Address
    def validIPAddress(self, IP: str) -> str:
        if "." in IP:
            # val ipv4
            ips = IP.split(".")
            if len(ips) != 4:
                return "Neither"
            for a in ips:
                try:
                    if a.startswith("0") and len(a) != 1:
                        return "Neither"
                    elif int(a) < 0 or int(a) > 255:
                        return "Neither"
                except:
                    return "Neither"
            return "IPv4"
        elif ":" in IP:
            ips = IP.split(":")
            if len(ips) != 8:
                return "Neither"
            # val ipv6
            for a in IP.split(":"):
                if len(a) == 0 or len(a) > 4:
                    return "Neither"
                for aa in a:
                    if aa not in "0123456789abcdefABCDEF":
                        return "Neither"
                # if a.startswith('00'):
                #     return "Neither"
            return "IPv6"


    # 480. Sliding Window Median
    def medianSlidingWindow(self, nums: List[int], k: int) -> List[float]:
        def move(h1, h2):
            x, i = heapq.heappop(h1)
            heapq.heappush(h2, (-x, i))
        def findMedian(h1, h2, k):
            if k % 2 == 1:
                return h2[0][0]
            else:
                return (h2[0][0] - h1[0][0]) / 2
        min_h, max_h = [], []
        for i, num in enumerate(nums[:k]):
            heapq.heappush(max_h, (-num, i))
        for i in range(k - k // 2):
            move(max_h, min_h)
        ans = [findMedian(max_h, min_h, k)]
        for i, num in enumerate(nums[k:]):
            if num >= min_h[0][0]:
                heapq.heappush(min_h, (num, i + k))
                if nums[i] <= min_h[0][0]:
                    move(min_h, max_h)
            else:
                heapq.heappush(max_h, (-num, i + k))
                if nums[i] >= min_h[0][0]:
                    move(max_h, min_h)
            while max_h and max_h[0][1] <= i:
                heapq.heappop(max_h)
            while min_h and min_h[0][1] <= i:
                heapq.heappop(min_h)
            ans.append(findMedian(max_h, min_h, k))
        return ans


    # 482. License Key Formatting
    def licenseKeyFormatting(self, S: str, K: int) -> str:
        # O(n)
        S = S.replace("-", "")
        first = len(S) % K
        new = ""

        for i, v in enumerate(S):
            new = new + v.upper()
            if i + 1 == first:
                new = new + "-"
            elif (i + 1 - first) % K == 0:
                new = new + "-"
        return new[:-1]


    # 494. Target Sum
    def findTargetSumWays(self, nums: List[int], S: int) -> int:
        sum_ = sum(nums)
        if S < -sum_ or S > sum_:
            return 0
        prev = {0: 1}
        for i in range(len(nums)):
            cur = collections.defaultdict(int)
            for val in prev:
                cur[val - nums[i]] += prev[val]
                cur[val + nums[i]] += prev[val]
            prev = cur
        return prev[S] if S in prev else 0


    # 498. Diagonal Traverse
    def findDiagonalOrder(self, matrix: List[List[int]]) -> List[int]:
        if not matrix:
            return []
        m, n = len(matrix), len(matrix[0])
        tmp = [[] for _ in range(m + n)]
        for i in range(m):
            for j in range(n):
                tmp[i + j].append(matrix[i][j])
        ans = []
        for i in range(m + n):
            if i % 2 == 1:
                ans.extend(tmp[i])
            else:
                ans.extend(tmp[i][::-1])
        return ans


    # 509. Fibonacci Number
    def fib(self, N: int) -> int:
        a, b = 0, 1
        for i in range(N):
            a, b = b, a + b
        return a


    # 515. Find Largest Value in Each Tree Row
    def largestValues(self, root: TreeNode) -> List[int]:
        if not root:
            return []
        cur = [root]
        ans = []
        while cur:
            next = []
            row_max = float("-inf")
            for node in cur:
                row_max = max(row_max, node.val)
                if node.left:
                    next.append(node.left)
                if node.right:
                    next.append(node.right)
            ans.append(row_max)
            cur = next
        return ans


    # 523. Continuous Subarray Sum
    def checkSubarraySum(self, nums: List[int], k: int) -> bool:
        dic = {0: -1}
        mod = 0
        for i, num in enumerate(nums):
            mod = (mod + num) % k if k != 0 else mod + num
            if mod in dic:
                if i - dic[mod] >= 2:
                    return True
            else:
                dic[mod] = i
        return False


    # 525. Contiguous Array
    def findMaxLength(self, nums: List[int]) -> int:
        dict = {0: -1}
        cur_score = 0
        ans = 0
        for i, num in enumerate(nums):
            cur_score += 1 if num == 1 else -1
            if cur_score not in dict:
                dict[cur_score] = i
            else:
                ans = max(ans, i - dict[cur_score])
        return ans

     
    # 538. Convert BST to Greater Tree
    def convertBST(self, root: TreeNode) -> TreeNode:
        nums = []
        def visit(root):
            if not root:
                return
            visit(root.right)
            nums.append(root.val)
            visit(root.left)
        visit(root)
        cum_nums = []
        cum_num = 0
        for num in nums:
            cum_num += num
            cum_nums.append(cum_num)
        def remark(root):
            if not root:
                return
            remark(root.left)
            root.val = cum_nums.pop()
            remark(root.right)
        remark(root)
        return root


    # 540. Single Element in a Sorted Array
    def singleNonDuplicate(self, nums: List[int]) -> int:
        left, right = 0, len(nums)
        while left < right:
            mid = left + (right - left) // 2
            if mid % 2 == 1:
                mid -= 1
            if mid + 1 < len(nums) and nums[mid] == nums[mid + 1]:
                left = mid + 2
            else:
                right = mid
        return nums[left]


    # 542. 01 Matrix
    def updateMatrix(self, matrix: List[List[int]]) -> List[List[int]]:
        m = len(matrix)
        n = len(matrix[0])
        d = [[float("inf") for _ in range(n)] for _ in range(m)]
        for i in range(m):
            for j in range(n):
                if matrix[i][j]:
                    if i > 0:
                        d[i][j] = min(d[i][j], d[i - 1][j] + 1)
                    if j > 0:
                        d[i][j] = min(d[i][j], d[i][j - 1] + 1)
                else:
                    d[i][j] = 0
        for i in range(m - 1, -1, -1):
            for j in range(n - 1, -1, -1):
                if matrix[i][j]:
                    if i < m - 1:
                        d[i][j] = min(d[i][j], d[i + 1][j] + 1)
                    if j < n - 1:
                        d[i][j] = min(d[i][j], d[i][j + 1] + 1)
                else:
                    d[i][j] = 0
        return d


    # 543. Diameter of Binary Tree
    def diameterOfBinaryTree(self, root: TreeNode) -> int:
        self.ans = 0
        def helper(root):  # return diameter ending at its parent
            if not root:
                return 0
            left = helper(root.left)
            right = helper(root.right)
            self.ans = max(self.ans, left + right)
            return max(left, right) + 1
        helper(root)
        return self.ans


    # 547. Number of Provinces
    def findCircleNum(self, M: List[List[int]]) -> int:
        q = collections.deque()
        ans = 0
        n = len(M)
        visited = [False] * n
        for i in range(n):
            if not visited[i]:
                q.append(i)
                visited[i] = True
                while q:
                    j = q.popleft()
                    for k in range(n):
                        if not visited[k] and M[j][k] == 1:
                            q.append(k)
                            visited[k] = True
                ans += 1
        return ans


    # 556. Next Greater Element III
    def nextGreaterElement(self, n: int) -> int:
        nums = list(str(n))
        m = len(nums)
        pivot = -1
        for i in range(m - 1, 0, -1):
            if nums[i - 1] < nums[i]:
                pivot = i - 1
                break
        if pivot == -1:
            return -1
        for i in range(m - 1, -1, -1):
            if nums[i] > nums[pivot]:
                nums[i], nums[pivot] = nums[pivot], nums[i]
                break
        i, j = pivot + 1, m - 1
        while i < j:
            nums[i], nums[j] = nums[j], nums[i]
            i += 1
            j -= 1
        n = int("".join(map(str, nums)))
        return -1 if n >= 2 ** 31 - 1 else n


    # 560. Subarray Sum Equals K
    def subarraySum(self, nums: List[int], k: int) -> int:
        cumsum = ans = 0
        dict = collections.defaultdict(int)
        for num in nums:
            cumsum += num
            if cumsum == k:
                ans += 1
            ans += dict[cumsum - k]
            dict[cumsum] += 1
        return ans


    # 567. Permutation in String
    def checkInclusion(self, s1: str, s2: str) -> bool:
        n1, n2 = len(s1), len(s2)
        counter1 = collections.Counter(s1)
        counter2 = collections.Counter(s2[:n1])
        for i in range(n1, n2):
            if counter1 == counter2:
                return True
            if s2[i] in counter2:
                counter2[s2[i]] += 1
            else:
                counter2[s2[i]] = 1
            counter2[s2[i - n1]] -= 1
            if counter2[s2[i - n1]] == 0:
                del counter2[s2[i - n1]]
        return counter1 == counter2


    # 572. Subtree of Another Tree
    def isSubtree(self, s: TreeNode, t: TreeNode) -> bool:
        def compare(node1, node2):
            if not node1 and not node2:
                return True
            if node1 and node2:
                return (
                    node1.val == node2.val
                    and compare(node1.left, node2.left)
                    and compare(node1.right, node2.right)
                )
            return False
        if not s:
            return False
        return compare(s, t) or self.isSubtree(s.left, t) or self.isSubtree(s.right, t)


    # 593. Valid Square
    def validSquare(
        self, p1: List[int], p2: List[int], p3: List[int], p4: List[int]
    ) -> bool:
        d = [
            (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2,
            (p1[0] - p3[0]) ** 2 + (p1[1] - p3[1]) ** 2,
            (p1[0] - p4[0]) ** 2 + (p1[1] - p4[1]) ** 2,
            (p2[0] - p3[0]) ** 2 + (p2[1] - p3[1]) ** 2,
            (p2[0] - p4[0]) ** 2 + (p2[1] - p4[1]) ** 2,
            (p3[0] - p4[0]) ** 2 + (p3[1] - p4[1]) ** 2,
        ]
        d = Counter(d)
        d = {val: key for key, val in d.items()}
        if len(d) != 2 or set(d.keys()) != set([2, 4]):
            return False
        return d[2] == 2 * d[4]


    # 599. Minimum Index Sum of Two Lists
    def findRestaurant(self, list1: List[str], list2: List[str]) -> List[str]:
        dict1 = {key: index for index, key in enumerate(list1)}
        dict2 = {key: index for index, key in enumerate(list2)}
        res = []
        min_sum = float("inf")
        for key in dict1:
            if key in dict2:
                index_sum = dict1[key] + dict2[key]
                if index_sum < min_sum:
                    res = [key]
                    min_sum = index_sum
                elif index_sum == min_sum:
                    res.append(key)
        return res


    # 605. Can Place Flowers
    def canPlaceFlowers(self, flowerbed: List[int], n: int) -> bool:
        m = len(flowerbed)
        for i in range(len(flowerbed)):
            if (
                flowerbed[i] == 0
                and (i == 0 or flowerbed[i - 1] == 0)
                and (i == m - 1 or flowerbed[i + 1] == 0)
            ):
                flowerbed[i] = 1
                n -= 1
                if n <= 0:
                    return True
        return n <= 0


    # 617. Merge Two Binary Trees
    def mergeTrees(self, t1: TreeNode, t2: TreeNode) -> TreeNode:
        if not t1 and not t2:
            return
        if t1 and t2:
            t1.val += t2.val
            t1.left = self.mergeTrees(t1.left, t2.left)
            t1.right = self.mergeTrees(t1.right, t2.right)
            return t1
        if t1 and not t2:
            return t1
        if not t1 and t2:
            return t2


    # 621. Task Scheduler
    def leastInterval(self, tasks: List[str], n: int) -> int:
        counter = collections.Counter(tasks)
        top_freq = counter.most_common(1)[0][1]
        top_tasks = len([key for key, val in counter.items() if val == top_freq])
        return max(len(tasks), (n + 1) * (top_freq - 1) + top_tasks)


    # 636. Exclusive Time of Functions
    def exclusiveTime(self, n: int, logs: List[str]) -> List[int]:
        prev = 0
        res = [0] * n
        stack = []
        for log in logs:
            idx, event, time = log.split(":")
            idx, time = int(idx), int(time)
            if event == "start":
                if len(stack):
                    res[stack[-1]] += time - prev
                stack.append(idx)
                prev = time
            else:
                res[stack[-1]] += time - prev + 1
                stack.pop()
                prev = time + 1
        return res


    # 637. Average of Levels in Binary Tree
    def averageOfLevels(self, root: TreeNode) -> List[float]:
        ans = []
        cur = [root]
        while cur:
            sum = 0
            next = []
            for node in cur:
                sum += node.val
                if node.left:
                    next.append(node.left)
                if node.right:
                    next.append(node.right)
            ans.append(sum / len(cur))
            cur = next
        return ans


    # 645. Set Mismatch
    def findErrorNums(self, nums: List[int]) -> List[int]:
        n = len(nums)
        for i in range(n):
            index = abs(nums[i]) - 1
            if nums[index] < 0:
                dup = abs(nums[i])
            else:
                nums[index] *= -1
        for i in range(n):
            if nums[i] > 0:
                missing = i + 1
                break
        return [dup, missing]


    # 647. Palindromic Substrings
    def countSubstrings(self, s: str) -> int:
        n = len(s)
        ans = 0
        for i in range(n):
            left = right = i
            while left >= 0 and right < n:
                if s[left] == s[right]:
                    ans += 1
                    left -= 1
                    right += 1
                else:
                    break
            left = i
            right = i + 1
            while left >= 0 and right < n:
                if s[left] == s[right]:
                    ans += 1
                    left -= 1
                    right += 1
                else:
                    break
        return ans


    # 653. Two Sum IV - Input is a BST
    def findTarget(self, root: TreeNode, k: int) -> bool:
        def inorder(root, nums):
            if not root:
                return
            inorder(root.left, nums)
            nums.append(root.val)
            inorder(root.right, nums)
        self.nums = []
        self.inorder(root)
        i, j = 0, len(self.nums) - 1
        while i < j:
            if self.nums[i] + self.nums[j] < k:
                i += 1
            elif self.nums[i] + self.nums[j] > k:
                j -= 1
            else:
                return True
        return False


    # 658. Find K Closest Elements
    def findClosestElements(self, arr: List[int], k: int, x: int) -> List[int]:
        l, r = 0, len(arr) - k
        while l < r:
            mid = l + (r - l) // 2
            if x - arr[mid] > arr[mid + k] - x:
                l = mid + 1
            else:
                r = mid
        return arr[l : l + k]


    # 665. Non-decreasing Array
    def checkPossibility(self, nums: List[int]) -> bool:
        p = None
        for i in range(len(nums) - 1):
            if nums[i] > nums[i + 1]:
                if p is not None:
                    return False
                p = i
        return (
            p is None
            or p == 0
            or p == len(nums) - 2
            or nums[p - 1] <= nums[p + 1]
            or nums[p] <= nums[p + 2]
        )


    # 670. Maximum Swap
    def maximumSwap(self, num: int) -> int:
        if num < 10:
            return num
        num = list(str(num))
        n = len(num)
        for i in range(1, n):
            if num[i - 1] < num[i]:
                break
        if i == n - 1:
            return int("".join(num))
        tmp_idx = n - 1
        for j in range(n - 1, i - 1, -1):
            if num[j] > num[tmp_idx]:
                tmp_idx = j
        num[j - 1], num[tmp_idx] = num[tmp_idx], num[j - 1]
        return int("".join(num))


    # 671. Second Minimum Node In a Binary Tree
    def findSecondMinimumValue(self, root: TreeNode) -> int:
        def helper(root):
            if not root:
                return
            if root.val < self.min:
                self.min = root.val
            elif root.val > self.min and root.val < self.second_min:
                self.second_min = root.val
            helper(root.left)
            helper(root.right)
        self.min = float("inf")
        self.second_min = float("inf")
        helper(root)
        return self.second_min if self.second_min != float("inf") else -1


    # 674. Longest Continuous Increasing Subsequence
    def findLengthOfLCIS(self, nums: List[int]) -> int:
        if len(nums) <= 1:
            return len(nums)
        res = curr = 1
        for i in range(1, len(nums)):
            if nums[i] > nums[i - 1]:
                curr += 1
            else:
                curr = 1
            res = max(res, curr)
        return res


    # 678. Valid Parenthesis String
    def checkValidString(self, s: str) -> bool:
        lo = hi = 0
        for char in s:
            if char == "(":
                lo += 1
                hi += 1
            elif char == ")":
                lo -= 1
                hi -= 1
            else:
                lo -= 1
                hi += 1
            if hi < 0:
                return False
            lo = max(0, lo)
        return lo == 0


    # 680. Valid Palindrome II
    def validPalindrome(self, s: str) -> bool:
        def isPanlidrome(i, j):
            l, r = i, j
            while l < r:
                if s[l] != s[r]:
                    return False
                l += 1
                r -= 1
            return True
        l, r = 0, len(s) - 1
        while l < r:
            if s[l] != s[r]:
                return isPanlidrome(l + 1, r) or isPanlidrome(l, r - 1)
            l += 1
            r -= 1
        return True


    # 686. Repeated String Match
    def repeatedStringMatch(self, a: str, b: str) -> int:
        times = len(b) // len(a)
        for i in range(3):
            if (a * (times + i)).find(b) >= 0:
                return times + i
        return -1


    # 689. Maximum Sum of 3 Non-Overlapping Subarrays
    def maxSumOfThreeSubarrays(self, nums: List[int], k: int) -> List[int]:
        W = []
        cur_sum = 0
        for i, num in enumerate(nums):
            cur_sum += num
            if i >= k:
                cur_sum -= nums[i - k]
            if i >= k - 1:
                W.append(cur_sum)
        left = [0] * len(W)
        best = 0
        for i in range(len(W)):
            if W[i] > W[best]:
                best = i
            left[i] = best
        right = [0] * len(W)
        best = len(W) - 1
        for i in range(len(W) - 1, -1, -1):
            if W[i] >= W[best]:  # for lexicographical order
                best = i
            right[i] = best
        ans = None
        for j in range(k, len(W) - k):
            i, l = left[j - k], right[j + k]
            if ans is None or W[i] + W[j] + W[l] > W[ans[0]] + W[ans[1]] + W[ans[2]]:
                ans = i, j, l
        return ans


    # 692. Top K Frequent Words
    def topKFrequent(self, words: List[str], k: int) -> List[str]:
        count = collections.Counter(words)
        heap = [(-freq, word) for word, freq in count.items()]
        heapq.heapify(heap)
        ans = []
        for i in range(k):
            ans.append(heapq.heappop(heap)[1])
        return ans


    # 695. Max Area of Island
    def maxAreaOfIsland(self, grid: List[List[int]]) -> int:
        m, n = len(grid), len(grid[0])
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        def bfs(i, j):
            area = 0
            q = collections.deque([(i, j)])
            grid[i][j] = 2
            while q:
                i, j = q.popleft()
                area += 1
                for d in directions:
                    next_i, next_j = i + d[0], j + d[1]
                    if (
                        0 <= next_i < m
                        and 0 <= next_j < n
                        and grid[next_i][next_j] == 1
                    ):
                        q.append((next_i, next_j))
                        grid[next_i][next_j] = 2
            return area
        ans = 0
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 1:
                    ans = max(ans, bfs(i, j))
        return ans


    # 709. To Lower Case
    def toLowerCase(self, str: str) -> str:
        return str.lower()
     

    # 721. Accounts Merge    
    def accountsMerge(self, accounts: List[List[str]]) -> List[List[str]]:
        email_to_name = {}
        graph = collections.defaultdict(set)
        for account in accounts:
            name, emails = account[0], account[1:]
            for email in emails:
                email_to_name[email] = name
                graph[email].add(emails[0])
                graph[emails[0]].add(email)
        visited = set()
        ans = []
        for email in graph:
            if email not in visited:
                person = []
                q = collections.deque()
                q.append(email)
                visited.add(email)
                person.append(email)
                while q:
                    node = q.popleft()
                    for nbr in graph[node]:
                        if nbr not in visited:
                            q.append(nbr)
                            visited.add(nbr)
                            person.append(nbr)
                person = [email_to_name[email]] + sorted(person)
                ans.append(person)
        return ans


    # 722. Remove Comments
    def removeComments(self, source: List[str]) -> List[str]:
        in_block = False
        ans = []
        for line in source:
            i = 0
            if not in_block:
                newline = []
            while i < len(line):
                if line[i : i + 2] == "/*" and not in_block:
                    in_block = True
                    i += 1
                elif line[i : i + 2] == "*/" and in_block:
                    in_block = False
                    i += 1
                elif not in_block and line[i : i + 2] == "//":
                    break
                elif not in_block:
                    newline.append(line[i])
                i += 1
            if newline and not in_block:
                ans.append("".join(newline))
        return ans


    # 724. Find Pivot Index
    def pivotIndex(self, nums: List[int]) -> int:
        sum_ = sum(nums)
        cum_sum = 0
        for i, num in enumerate(nums):
            cum_sum += num
            if 2 * (cum_sum - num) + num == sum_:
                return i
        return -1


    # 739. Daily Temperatures
    def dailyTemperatures(self, T: List[int]) -> List[int]:
        stack = []
        n = len(T)
        ans = [0] * n
        for i in range(n - 1, -1, -1):
            while stack and T[i] >= T[stack[-1]]:
                stack.pop()
            if stack:
                ans[i] = stack[-1] - i
            stack.append(i)
        return ans


    # 743. Network Delay Time
    def networkDelayTime(self, times: List[List[int]], N: int, K: int) -> int:
        dic = collections.defaultdict(dict)
        for u,v,w in times:
            dic[u][v] = w 
        q = [[0, K]]
        visited = set([K])
        while q:           
            l, end = heapq.heappop(q)
            visited.add(end)
            if len(visited) == N: 
                return l
            print(l)
            for nextn in dic[end]:
                if nextn not in visited:                    
                    heapq.heappush(q, [l + dic[end][nextn], nextn])
        return -1


    # 746. Min Cost Climbing Stairs
    def minCostClimbingStairs(self, cost: List[int]) -> int:
        res = []
        for i in range(len(cost)):
            if i == 0 or i == 1:
                res.append(cost[i])
            else:
                res.append(min(res[-1], res[-2]) + cost[i])
        return min(res[-1], res[-2])


    # 752. Open the Lock
    def openLock(self, deadends: List[str], target: str) -> int:
        deadends = set(deadends)
        if "0000" in deadends:
            return -1
        q = collections.deque([["0000", 0]])
        visited = {"0000"}

        while q:
            node, depth = q.popleft()
            if node == target:
                return depth
            for i in range(4):
                x = int(node[i])
                for d in (-1, 1):
                    y = (x + d) % 10
                    new_node = node[:i] + str(y) + node[i + 1 :]
                    if new_node not in visited and new_node not in deadends:
                        visited.add(new_node)
                        q.append([new_node, depth + 1])
        return -1


    # 753. Cracking the Safe
    def crackSafe(self, n, k):
        st = "0" * n
        if n == 1:
            return "".join(map(str, range(k)))
        seen = set()
        seen.add("0" * n)
        while len(seen) < k ** n:
            for num in map(str, range(k - 1, -1, -1)):
                if (st[-n + 1 :] + num) not in seen:
                    seen.add(st[-n + 1 :] + num)
                    st = st + num
                    break
        return st


    # 763. Partition Labels
    def partitionLabels(self, S: str) -> List[int]:
        last_idx = {}
        for i, char in enumerate(S):
            last_idx[char] = i
        ans = []
        l = -1
        r = 0
        for i, char in enumerate(S):
            r = max(r, last_idx[char])
            if r == i:
                ans.append(r - l)
                l = i
        return ans


    # 766. Toeplitz Matrix
    def isToeplitzMatrix(self, matrix: List[List[int]]) -> bool:
        ans = True
        m, n = len(matrix), len(matrix[0])
        for j in range(n):
            for k in range(min(m, n - j)):
                if matrix[k][j + k] != matrix[0][j]:
                    return False
        for i in range(m):
            for k in range(min(m - i, n)):
                if matrix[i + k][k] != matrix[i][0]:
                    return False
        return True


    # 767. Reorganize String
    def reorganizeString(self, S: str) -> str:
        heap = [(-c, w) for w, c in Counter(S).items()]
        heapq.heapify(heap)
        ans = ""
        while len(heap) > 1:
            c1, w1 = heapq.heappop(heap)
            c2, w2 = heapq.heappop(heap)
            ans += w1
            ans += w2
            if -c1 > 1:
                heapq.heappush(heap, (c1 + 1, w1))
            if -c2 > 1:
                heapq.heappush(heap, (c2 + 1, w2))
        if heap:
            c, w = heap[0]
            if c < -1:
                return ""
            else:
                ans += heap[0][1]
        return ans


    # 771. Jewels and Stones
    def numJewelsInStones(self, J: str, S: str) -> int:
        J_set = set(J)
        res = 0
        for s in S:
            if s in J_set:
                res += 1
        return res


    # 785. Is Graph Bipartite?
    def isBipartite(self, graph: List[List[int]]) -> bool:
        colors = {}
        q = deque()
        for i in range(len(graph)):
            if i not in colors:
                colors[i] = 0
                q.append(i)
                while q:
                    j = q.popleft()
                    for k in graph[j]:
                        if k in colors:
                            if colors[k] == colors[j]:
                                return False
                        else:
                            colors[k] = colors[j] ^ 1
                            q.append(k)
        return True


    # 787. Cheapest Flights Within K Stops
    def findCheapestPrice(
        self, n: int, flights: List[List[int]], src: int, dst: int, K: int
    ) -> int:
        f = collections.defaultdict(dict)
        for a, b, p in flights:
            f[a][b] = p
        h = [(0, src, K + 1)]
        while h:
            p, i, k = heapq.heappop(h)
            if i == dst:
                return p
            if k > 0:
                for j in f[i]:
                    heapq.heappush(h, (p + f[i][j], j, k - 1))
        return -1


    # 791. Custom Sort String
    def customSortString(self, S: str, T: str) -> str:
        counter = collections.Counter(T)
        ans = []
        for s in S:
            if s in counter:
                ans.extend([s] * counter[s])
                del counter[s]
        for s in counter:
            ans.append([s] * counter[s])
        return "".join(ans)


    # 797. All Paths From Source to Target
    def allPathsSourceTarget(self, graph: List[List[int]]) -> List[List[int]]:
        ans = []
        n = len(graph)
        def helper(index, path):
            if index == n - 1:
                ans.append(path[:])
                return
            for node in graph[index]:
                path.append(node)
                helper(node, path)
                path.pop()
        helper(0, [0])
        return ans


    # 809. Expressive Words
    def expressiveWords(self, S: str, words: List[str]) -> int:
        if S == "":
            return 0
        w, count_s = zip(*[(c, len(list(v))) for c, v in itertools.groupby(S)])
        ans = 0
        for word in words:
            if word == "":
                continue
            w_w, count_w = zip(*[(c, len(list(v))) for c, v in itertools.groupby(word)])
            if w != w_w:
                continue
            ans += all(
                [c1 == c2 or c1 >= max(c2, 3) for c1, c2 in zip(count_s, count_w)]
            )
        return ans


    # 819. Most Common Word
    def mostCommonWord(self, paragraph: str, banned: List[str]) -> str:
        paragraph += " "
        count = defaultdict(int)
        banned_set = set(banned)
        word = ""
        most_common_word = None
        most_common_count = 0
        for char in paragraph:
            if not char.isalpha():
                if word and word not in banned_set:
                    count[word] += 1
                    if count[word] > most_common_count:
                        most_common_count = count[word]
                        most_common_word = word
                word = ""
            else:
                word += char.lower()
        return most_common_word


    # 824. Goat Latin
    def toGoatLatin(self, S: str) -> str:
        words = S.split(" ")
        for i in range(len(words)):
            if words[i][0].lower() in "aeiou":
                words[i] += "ma"
            else:
                words[i] = words[i][1:] + words[i][0] + "ma"
            words[i] += "a" * (i + 1)
        return " ".join(words)


    # 825. Friends Of Appropriate Ages
    def numFriendRequests(self, ages: List[int]) -> int:
        counter = Counter(ages)
        ans = 0
        for ageA, countA in counter.items():
            for ageB, countB in counter.items():
                if ageB <= ageA * 0.5 + 7:
                    continue
                if ageB > ageA:
                    continue
                ans += countA * countB
                if ageA == ageB:
                    ans -= countA
        return ans


    # 827. Making A Large Island
    def largestIsland(self, grid: List[List[int]]) -> int:
        n = len(grid)
        directions = [[-1, 0], [1, 0], [0, 1], [0, -1]]
        def findArea(i, j, index):
            ans = 1
            grid[i][j] = index
            for d in directions:
                next_i, next_j = i + d[0], j + d[1]
                if 0 <= next_i < n and 0 <= next_j < n and grid[next_i][next_j] == 1:
                    ans += findArea(next_i, next_j, index)
            return ans
        area = {}
        index = 2
        for i in range(n):
            for j in range(n):
                if grid[i][j] == 1:
                    area[index] = findArea(i, j, index)
                    index += 1
        ans = max(area.values()) if area else 1
        for i in range(n):
            for j in range(n):
                if grid[i][j] == 0:
                    seen = set()
                    for d in directions:
                        next_i, next_j = i + d[0], j + d[1]
                        if (
                            0 <= next_i < n
                            and 0 <= next_j < n
                            and grid[next_i][next_j] > 1
                        ):
                            seen.add(grid[next_i][next_j])
                    curr_ans = 1 + sum(area[index] for index in seen) if seen else 0
                    ans = max(ans, curr_ans)
        return ans


    # 839. Similar String Groups
    def numSimilarGroups(self, A: List[str]) -> int:
        def similar(word1, word2):
            diff = 0
            for i in range(len(word)):
                if word1[i] != word2[i]:
                    diff += 1
                if diff > 2:
                    return False
            return True
        def dfs(word):
            for word2 in A:
                if word2 not in visited and similar(word, word2):
                    visited.add(word2)
                    dfs(word2)
        group = 0
        visited = set()
        for word in A:
            if word not in visited:
                visited.add(word)
                dfs(word)
                group += 1
        return group


    # 844. Backspace String Compare
    def backspaceCompare(self, S: str, T: str) -> bool:
        def f(S):
            skip = 0
            for s in reversed(S):
                if s == "#":
                    skip += 1
                elif skip > 0:
                    skip -= 1
                else:
                    yield s
            yield ""
        for x, y in zip(f(S), f(T)):
            if x != y:
                return False
        return True


    # 852. Peak Index in a Mountain Array
    def peakIndexInMountainArray(self, arr: List[int]) -> int:
        left, right = 1, len(arr) - 2
        while left <= right:
            mid = left + (right - left) // 2
            if arr[mid] > arr[mid - 1] and arr[mid] > arr[mid + 1]:
                return mid
            elif arr[mid] <= arr[mid - 1]:
                right = mid - 1
            else:
                left = mid + 1


    # 858. Mirror Reflection
    def mirrorReflection(self, p: int, q: int) -> int:
        g = gcd(p, q)
        p /= g
        q /= g
        p %= 2
        q %= 2
        if p == 1 and q == 1:
            return 1
        if p == 0 and q == 1:
            return 2
        if p == 1 and q == 0:
            return 0


    # 863. All Nodes Distance K in Binary Tree
    def distanceK(self, root: TreeNode, target: TreeNode, K: int) -> List[int]:
        # build graph
        graph = collections.defaultdict(set)
        def traverse(root):
            if not root:
                return
            if root.left:
                graph[root.val].add(root.left.val)
                graph[root.left.val].add(root.val)
            if root.right:
                graph[root.val].add(root.right.val)
                graph[root.right.val].add(root.val)
            traverse(root.left)
            traverse(root.right)
        traverse(root)
        # BFS
        q = collections.deque()
        q.append(target.val)
        visited = set([target.val])
        while K > 0:
            n = len(q)
            for _ in range(n):
                val = q.popleft()
                for nbr in graph[val]:
                    if nbr not in visited:
                        q.append(nbr)
                        visited.add(nbr)
            K -= 1
        return list(q)


    # 875. Koko Eating Bananas
    def minEatingSpeed(self, piles: List[int], H: int) -> int:
        def possible(K):
            return sum((p - 1) // K + 1 for p in piles) <= H

        l, r = 1, max(piles)
        while l < r:
            mid = l + (r - l) // 2
            if not possible(mid):
                l = mid + 1
            else:
                r = mid
        return l


    # 896. Monotonic Array
    def isMonotonic(self, A: List[int]) -> bool:
        dirc = True
        for i in range(1, len(A)):
            if A[i] < A[i - 1]:
                pass1 = False
                break
        pass2 = True
        for i in range(1, len(A)):
            if A[i] > A[i - 1]:
                pass2 = False
                break
        return pass1 or pass2


    # 904. Fruit Into Baskets
    def totalFruit(self, tree):
        i = 0
        ans = 0
        basket = dict()
        for j in range(len(tree)):
            if tree[j] in basket.keys():
                basket[tree[j]] += 1
            else:
                basket[tree[j]] = 1
                while len(basket.keys()) > 2:
                    basket[tree[i]] -= 1
                    if basket[tree[i]] == 0:
                        del basket[tree[i]]
                    i += 1
            ans = max(ans, j - i + 1)
        return ans


    # 905. Sort Array By Parity
    def sortArrayByParity(self, A: List[int]) -> List[int]:
        ans = [0] * len(A)
        i = 0
        j = len(A) - 1
        for num in A:
            if num % 2 == 0:
                ans[i] = num
                i += 1
            else:
                ans[j] = num
                j -= 1
        if i - j == 1:
            return ans
        else:
            return [-1]


    # 912. Sort an Array
    def sortArray(self, nums: List[int]) -> List[int]:
        def partition(left, right):
            idx = random.randint(left, right)
            nums[idx], nums[right] = nums[right], nums[idx]
            i = left
            for j in range(left, right):
                if nums[j] < nums[right]:
                    nums[i], nums[j] = nums[j], nums[i]
                    i += 1
            nums[i], nums[right] = nums[right], nums[i]
            return i
        def quickSort(left, right):
            if left < right:
                pivot = partition(left, right)
                quickSort(left, pivot - 1)
                quickSort(pivot + 1, right)
        quickSort(0, len(nums) - 1)
        return nums


    # 921. Minimum Add to Make Parentheses Valid
    def minAddToMakeValid(self, S: str) -> int:
        left = right = 0
        for char in S:
            if char == "(":
                left += 1
            else:
                if left:
                    left -= 1
                else:
                    right += 1
        return left + right


    # 929. Unique Email Addresses
    def numUniqueEmails(self, emails: List[str]) -> int:
        email_set = set()
        for email in emails:
            name, domain = email.split("@")
            name = name.split("+")[0].replace(".", "")
            email_set.add(name + "@" + domain)
        return len(email_set)


    # 934. Shortest Bridge
    def shortestBridge(self, A: List[List[int]]) -> int:
        m, n = len(A), len(A[0])
        directions = [[0, 1], [0, -1], [-1, 0], [1, 0]]
        # find island1
        def findOne():
            for i in range(m):
                for j in range(n):
                    if A[i][j] == 1:
                        A[i][j] = 2
                        return i, j
        i, j = findOne()
        island1 = collections.deque()
        q1 = collections.deque()
        island1.append((i, j, 0))
        q1.append((i, j))
        while q1:
            i, j = q1.popleft()
            for direction in directions:
                next_i, next_j = i + direction[0], j + direction[1]
                if 0 <= next_i < m and 0 <= next_j < n and A[next_i][next_j] == 1:
                    A[next_i][next_j] = 2
                    island1.append((next_i, next_j, 0))
                    q1.append((next_i, next_j))
        # expand to island2
        while island1:
            i, j, depth = island1.popleft()
            for direction in directions:
                next_i, next_j = i + direction[0], j + direction[1]
                if 0 <= next_i < m and 0 <= next_j < n:
                    if A[next_i][next_j] == 1:
                        return depth
                    if A[next_i][next_j] == 0:
                        A[next_i][next_j] = 2
                        island1.append((next_i, next_j, depth + 1))

    
    # 937. Reorder Data in Log Files
    def reorderLogFiles(self, logs: List[str]) -> List[str]:
        def get_key(log):
            _id, rest = log.split(" ", maxsplit=1)
            return (0, rest, _id) if rest[0].isalpha() else (1,)
        return sorted(logs, key=get_key)


    # 938. Range Sum of BST
    def rangeSumBST(self, root: TreeNode, L: int, R: int) -> int:
        self.ans = 0
        def helper(root, L, R):
            if not root or L > R:
                return
            if root.val > L:
                helper(root.left, L, R)
            if root.val < R:
                helper(root.right, L, R)
            if L <= root.val <= R:
                self.ans += root.val
        helper(root, L, R)
        return self.ans


    # 939. Minimum Area Rectangle
    def minAreaRect(self, points: List[List[int]]) -> int:
        n = len(points)
        nx = len(set([x for x, _ in points]))
        ny = len(set([y for _, y in points]))
        dict = collections.defaultdict(list)
        if nx > ny:
            for x, y in points:
                dict[x].append(y)
        else:
            for x, y in points:
                dict[y].append(x)
        visited = {}
        ans = float("inf")
        for x in sorted(dict):
            m = len(dict[x])
            dict[x].sort()
            for i in range(m):
                for j in range(i + 1, m):
                    y1, y2 = dict[x][i], dict[x][j]
                    if (y1, y2) in visited:
                        ans = min(ans, abs((y1 - y2) * (x - visited[(y1, y2)])))
                    visited[(y1, y2)] = x
        return ans if ans < float("inf") else 0


    # 947. Most Stones Removed with Same Row or Column
    def removeStones(self, stones: List[List[int]]) -> int:
        self.parent = range(20000)
        def find(x):
            if x != self.parent[x]:
                return find(self.parent[x])
            return x
        def union(x, y):
            xr = find(x)
            yr = find(y)
            self.parent[xr] = yr
        for x, y in stones:
            union(x, 10000 + y)
        return len(stones) - len(set([find(x) for x, y in stones]))


    # 951. Flip Equivalent Binary Trees
    def flipEquiv(self, root1, root2):
        if root1 is root2:
            return True
        if not root1 or not root2 or root1.val != root2.val:
            return False
        return (
            self.flipEquiv(root1.left, root2.left)
            and self.flipEquiv(root1.right, root2.right)
            or self.flipEquiv(root1.left, root2.right)
            and self.flipEquiv(root1.right, root2.left)
        )


    # 953. Verifying an Alien Dictionary
    def isAlienSorted(self, words: List[str], order: str) -> bool:
        order = {char: i for i, char in enumerate(order)}
        for word1, word2 in zip(words[:-1], words[1:]):
            n1, n2 = len(word1), len(word2)
            i = 0
            while i < min(n1, n2):
                if word1[i] == word2[i]:
                    i += 1
                    continue
                elif order[word1[i]] > order[word2[i]]:
                    return False
                else:
                    break
            else:
                if i < n1:
                    return False
        return True


    # 958. Check Completeness of a Binary Tree
    def isCompleteTree(self, root: TreeNode) -> bool:
        if not root:
            return True
        cur = [root]
        flag = False
        while cur:
            next = []
            for node in cur:
                if not node:
                    flag = True
                else:
                    if flag:
                        return False
                    next.append(node.left)
                    next.append(node.right)
            cur = next
        return True


    # 969. Pancake Sorting
    def pancakeSort(self, arr: List[int]) -> List[int]:
        ans = []
        for i in range(len(arr), 1, -1):
            j = arr.index(i)
            ans.extend([j + 1, len(arr)])
            arr = arr[:j:-1] + arr[:j]
        return ans


    # 973. K Closest Points to Origin
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        def dist(point):
            return point[0] ** 2 + point[1] ** 2
        def partition(left, right):
            idx = random.randint(left, right)
            points[idx], points[right] = points[right], points[idx]
            pivot = dist(points[right])
            i = j = left
            while j <= right:
                if dist(points[j]) < pivot:
                    points[j], points[i] = points[i], points[j]
                    i += 1
                j += 1
            points[i], points[right] = points[right], points[i]
            return i
        def sort(left, right):
            if left == right:
                return
            pivot = partition(left, right)
            if K == pivot:
                return
            elif K < pivot:
                sort(left, pivot - 1)
            elif K > pivot:
                sort(pivot + 1, right)
        sort(0, len(points) - 1)
        return points[:K]


    # 974. Subarray Sums Divisible by K
    def subarraysDivByK(self, A: List[int], K: int) -> int:
        cumsum = ans = 0
        dic = collections.defaultdict(int)
        for a in A:
            cumsum = (cumsum + a + K) % K
            if cumsum == 0:
                ans += 1
            ans += dic[cumsum]
            dic[cumsum] += 1
        return ans


    # 975. Odd Even Jump
    def oddEvenJumps(self, A: List[int]) -> int:
        def next_large(self, A):
            stack = []
            ans = [-1] * len(A)
            for fir, sec in sorted([(v, i) for i, v in enumerate(A)]):
                while stack and stack[-1] <= sec:
                    ans[stack.pop()] = sec
                stack.append(sec)
            return ans
        def next_small(self, A):
            stack = []
            ans = [-1] * len(A)
            for fir, sec in sorted([(-v, i) for i, v in enumerate(A)]):
                while stack and stack[-1] <= sec:
                    ans[stack.pop()] = sec
                stack.append(sec)
            return ans
        arr = [[0, 0] for i in range(len(A))]
        next_large = self.next_large(A)
        next_small = self.next_small(A)
        for i in range(len(A) - 1, -1, -1):
            if i == len(A) - 1:
                arr[i] = [1, 1]
                continue
            index = next_large[i]
            if index > 0 and arr[index][1] == 1:
                arr[i][0] = 1
            index = next_small[i]
            if index > 0 and arr[index][0] == 1:
                arr[i][1] = 1
        return sum([item[0] for item in arr])


    # 977. Squares of a Sorted Array
    def sortedSquares(self, A: List[int]) -> List[int]:
        i, j = 0, len(A) - 1
        res = []
        while i <= j:
            if abs(A[i]) < abs(A[j]):
                res.append(A[j] ** 2)
                j -= 1
            elif abs(A[i]) > abs(A[j]):
                res.append(A[i] ** 2)
                i += 1
            else:
                res.append(A[i] ** 2)
                if i < j:
                    res.append(A[j] ** 2)
                i += 1
                j -= 1
        return res[::-1]


    # 983. Minimum Cost For Tickets
    def mincostTickets(self, days: List[int], costs: List[int]) -> int:
        daysSet = set(days)
        durations = [1, 7, 30]
        memo = {}
        def dp(i):
            if i not in memo:
                if i > 365:
                    memo[i] = 0
                elif i in daysSet:
                    ans = float("inf")
                    for d, c in zip(durations, costs):
                        ans = min(ans, dp(i + d) + c)
                    memo[i] = ans
                else:
                    memo[i] = dp(i + 1)
            return memo[i]
        return dp(1)


    # 986. Interval List Intersections
    def intervalIntersection(
        self, A: List[List[int]], B: List[List[int]]
    ) -> List[List[int]]:
        ans = []
        i = j = 0
        m, n = len(A), len(B)
        while i < m and j < n:
            lo = max(A[i][0], B[j][0])
            hi = min(A[i][1], B[j][1])
            if lo <= hi:
                ans.append([lo, hi])
            if A[i][1] <= B[j][1]:
                i += 1
            else:
                j += 1
        return ans


    # 987. Vertical Order Traversal of a Binary Tree
    def verticalTraversal(self, root: TreeNode) -> List[List[int]]:
        if not root:
            return
        q = collections.deque()
        q.append((0, 0, root))
        l = [[0, 0, root.val]]
        while q:
            col, row, node = q.popleft()
            if node.left:
                q.append((col - 1, row + 1, node.left))
                l.append([col - 1, row + 1, node.left.val])
            if node.right:
                q.append((col + 1, row + 1, node.right))
                l.append([col + 1, row + 1, node.right.val])
        l.sort()
        ans = []
        cur_ans = []
        cur_col = l[0][0]
        for col, _, val in l:
            if col == cur_col:
                cur_ans.append(val)
            else:
                cur_col = col
                ans.append(cur_ans)
                cur_ans = [val]
        ans.append(cur_ans)
        return ans


    # 989. Add to Array-Form of Integer
    def addToArrayForm(self, A: List[int], K: int) -> List[int]:
        ans = []
        i = len(A) - 1
        carry = 0
        while i >= 0 and K > 0:
            k = K % 10
            tmp = carry + A[i] + k
            ans.append(tmp % 10)
            carry = tmp // 10
            K //= 10
            i -= 1
        while i >= 0:
            tmp = carry + A[i]
            ans.append(tmp % 10)
            carry = tmp // 10
            i -= 1
        while K > 0:
            k = K % 10
            tmp = carry + k
            ans.append(tmp % 10)
            carry = tmp // 10
            K //= 10
        if carry > 0:
            ans.append(1)
        return "".join(map(str, ans[::-1]))


    # 993. Cousins in Binary Tree
    def isCousins(self, root: TreeNode, x: int, y: int) -> bool:
        nodes = collections.defaultdict(set)
        q = collections.deque()
        q.append((root, 0, 0))
        while q:
            node, level, parent = q.popleft()
            nodes[node.val] = (level, parent)
            if node.left:
                q.append((node.left, level + 1, node.val))
            if node.right:
                q.append((node.right, level + 1, node.val))
        if nodes[x][0] == nodes[y][0] and nodes[x][1] != nodes[y][1]:
            return True
        return False


    # 994. Rotting Oranges
    def orangesRotting(self, grid: List[List[int]]) -> int:
        if not grid:
            return 0
        #  visited = [[ -1 for i in range(len(grid))] for j in range(len(grid[0]))]
        que = collections.deque()
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == 2:
                    grid[i][j] = 0
                    que.append((i, j, 0))
        dire = [[-1, 0], [0, 1], [0, -1], [1, 0]]
        level = 0
        while que:
            r, c, level = que.popleft()
            for i, j in dire:
                nr = r + i
                nc = c + j
                if nr >= 0 and nr < len(grid) and nc >= 0 and nc < len(grid[0]):
                    if grid[nr][nc] == 1:
                        grid[nr][nc] = 0
                        que.append((nr, nc, level + 1))
        return level if sum([i for row in grid for i in row]) == 0 else -1


    # 1004. Max Consecutive Ones III
    def longestOnes(self, A: List[int], K: int) -> int:
        ans = 0
        left = 0
        for right in range(len(A)):
            if A[right] == 0:
                K -= 1
            while K < 0:
                if A[left] == 0:
                    K += 1
                left += 1
            ans = max(ans, right - left + 1)
        return ans


    # 1026. Maximum Difference Between Node and Ancestor
    def maxAncestorDiff(self, root: TreeNode) -> int:
        def helper(root):
            if not root:
                return None, None
            cur_max = cur_min = root.val
            left_min, left_max = helper(root.left)
            right_min, right_max = helper(root.right)
            if left_min is not None:
                cur_min = min(cur_min, left_min)
                self.ans = max(self.ans, abs(root.val - left_min))
            if left_max is not None:
                cur_max = max(cur_max, left_max)
                self.ans = max(self.ans, abs(root.val - left_max))
            if right_min is not None:
                cur_min = min(cur_min, right_min)
                self.ans = max(self.ans, abs(root.val - right_min))
            if right_max is not None:
                cur_max = max(cur_max, right_max)
                self.ans = max(self.ans, abs(root.val - right_max))
            return cur_min, cur_max
        self.ans = float("-inf")
        helper(root)
        return self.ans


    # 1027. Longest Arithmetic Subsequence
    def longestArithSeqLength(self, A: List[int]) -> int:
        dp = {}
        for i in range(len(A)):
            for j in range(i + 1, len(A)):
                dp[j, A[j] - A[i]] = dp.get((i, A[j] - A[i]), 1) + 1
        return max(dp.values())


    # 1047. Remove All Adjacent Duplicates In String
    def removeDuplicates(self, S: str) -> str:
        stack = []
        for char in S:
            if len(stack) and stack[-1] == char:
                stack.pop()
            else:
                stack.append(char)
        return "".join(stack)


    # 1053. Previous Permutation With One Swap
    def prevPermOpt1(self, A: List[int]) -> List[int]:
        i = len(A) - 2
        while i >= 0 and A[i] <= A[i + 1]:
            i -= 1
        if i == -1:
            return A

        j = len(A) - 1
        while j > i and (A[j] >= A[i] or A[j] == A[j - 1]):
            j -= 1

        A[i], A[j] = A[j], A[i]
        return A


    # 1091. Shortest Path in Binary Matrix
    def shortestPathBinaryMatrix(self, grid: List[List[int]]) -> int:
        if not grid or grid[0][0] == 1:
            return -1
        m, n = len(grid), len(grid[0])
        q = collections.deque()
        q.append((0, 0, 1))
        grid[0][0] = 1
        while q:
            i, j, step = q.popleft()
            if i == m - 1 and j == n - 1:
                return step
            for d in (
                (0, 1),
                (0, -1),
                (1, 0),
                (-1, 0),
                (1, 1),
                (1, -1),
                (-1, 1),
                (-1, -1),
            ):
                next_i, next_j = i + d[0], j + d[1]
                if 0 <= next_i < m and 0 <= next_j < n and grid[next_i][next_j] == 0:
                    q.append((next_i, next_j, step + 1))
                    grid[next_i][next_j] = 1
        return -1


    # 1094. Car Pooling
    def carPooling(self, trips: List[List[int]], capacity: int) -> bool:
        deltas = [0] * 1001
        for num, start, end in trips:
            deltas[start] += num
            deltas[end] -= num
        cum_num = 0
        for delta in deltas:
            cum_num += delta
            if cum_num > capacity:
                return False
        return True


    # 1108. Defanging an IP Address
    def defangIPaddr(self, address: str) -> str:
        return "[.]".join(address.split("."))


    # 1123. Lowest Common Ancestor of Deepest Leaves
    def lcaDeepestLeaves(self, root: TreeNode) -> TreeNode:
        def helper(root):
            if not root:
                return 0, root
            left_height, left_lca = helper(root.left)
            right_height, right_lca = helper(root.right)
            if left_height < right_height:
                return right_height + 1, right_lca
            if left_height > right_height:
                return left_height + 1, left_lca
            return left_height + 1, root
        return helper(root)[1]


    # 1137. N-th Tribonacci Number
    def tribonacci(self, n: int) -> int:
        memo = {}
        def helper(n):
            if n == 0:
                return 0
            if n == 1 or n == 2:
                return 1
            if n not in memo:
                memo[n] = helper(n - 1) + helper(n - 2) + helper(n - 3)
            return memo[n]
        return helper(n)


    # 1143. Longest Common Subsequence
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        n1, n2 = len(text1), len(text2)
        dp = [[0] * (n2 + 1) for _ in range(n1 + 1)]
        for i in range(n1 - 1, -1, -1):
            for j in range(n2 - 1, -1, -1):
                if text1[i] == text2[j]:
                    dp[i][j] = 1 + dp[i + 1][j + 1]
                else:
                    dp[i][j] = max(dp[i + 1][j], dp[i][j + 1], dp[i + 1][j + 1])
        return dp[0][0]


    # 1162. As Far from Land as Possible
    def maxDistance(self, grid: List[List[int]]) -> int:
        m, n = len(grid), len(grid[0])
        q = collections.deque(
            [(i, j) for i in range(m) for j in range(n) if grid[i][j] == 1]
        )
        if len(q) == m * n or len(q) == 0:
            return -1
        level = -1
        while q:
            size = len(q)
            for _ in range(size):
                i, j = q.popleft()
                for d in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    next_i, next_j = i + d[0], j + d[1]
                    if (
                        0 <= next_i < m
                        and 0 <= next_j < n
                        and grid[next_i][next_j] == 0
                    ):
                        q.append((next_i, next_j))
                        grid[next_i][next_j] = 1
            level += 1
        return level


    # 1209. Remove All Adjacent Duplicates in String II
    def removeDuplicates2(self, s: str, k: int) -> str:
        stack = []
        for char in s:
            if not stack or stack[-1][0] != char:
                stack.append([char, 1])
            elif stack[-1][0] == char:
                stack.append([char, stack.pop()[-1] + 1])
                if stack[-1][1] == k:
                    stack.pop()
        return "".join([char * freq for char, freq in stack])


    # 1233. Remove Sub-Folders from the Filesystem
    def removeSubfolders(self, folder: List[str]) -> List[str]:
        trie = {}
        ans = []
        folder.sort()
        for path in folder:
            node = trie
            for x in path.split("/")[1:]:
                if x not in node:
                    node[x] = {}
                node = node[x]
                if "$" in node:
                    break
            else:  # successfully finish the loop
                node["$"] = True
                ans.append(path)
        return ans


    # 1239. Maximum Length of a Concatenated String with Unique Characters
    def maxLength(self, A):
        dp = [set()]
        for a in A:
            if len(set(a)) < len(a):
                continue
            a = set(a)
            for c in dp[:]:
                if a & c:
                    continue
                dp.append(a | c)
        return max(len(a) for a in dp)


    # 1249. Minimum Remove to Make Valid Parentheses
    def minRemoveToMakeValid(self, s: str) -> str:
        stack = []
        for i, ch in enumerate(s):
            if ch == "(":
                stack.append(("(", i))
            elif ch == ")":
                if not stack or stack[-1][0] != "(":
                    stack.append((")", i))
                else:
                    stack.pop()
        idx = set([i for _, i in stack])
        return "".join([ch for i, ch in enumerate(s) if i not in idx])


    # 1254. Number of Closed Islands
    def closedIsland(self, grid: List[List[int]]) -> int:
        if not grid:
            return 0
        m, n = len(grid), len(grid[0])
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        def bfs(i, j):
            q = collections.deque([(i, j)])
            while q:
                i, j = q.popleft()
                for d in directions:
                    next_i, next_j = i + d[0], j + d[1]
                    if (
                        0 <= next_i < m
                        and 0 <= next_j < n
                        and grid[next_i][next_j] == 0
                    ):
                        q.append((next_i, next_j))
                        grid[next_i][next_j] = 1
        # mark 0's connected to boarder with 2
        q = collections.deque()
        for i in range(m):
            for j in [0, n - 1]:
                if grid[i][j] == 0:
                    bfs(i, j)
        for j in range(n):
            for i in [0, m - 1]:
                if grid[i][j] == 0:
                    bfs(i, j)
        # now BFS to get closed islands
        ans = 0
        for i in range(1, m - 1):
            for j in range(1, n - 1):
                if grid[i][j] == 0:
                    bfs(i, j)
                    ans += 1
        return ans


    # 1262. Greatest Sum Divisible by Three
    def maxSumDivThree(self, nums: List[int]) -> int:
        ans = [0, 0, 0]
        n = len(nums)
        for i in range(n):
            for prev in ans[:]:
                mod = (prev + nums[i]) % 3
                ans[mod] = max(ans[mod], prev + nums[i])
        return ans[0]


    # 1269. Number of Ways to Stay in the Same Place After Some Steps
    def numWays(self, steps: int, arrLen: int) -> int:
        @lru_cache(None)
        def dfs(steps, idx):
            if idx < 0 or idx >= arrLen:
                return 0
            if steps == 0:
                return 1 * (idx == 0)
            return (
                dfs(steps - 1, idx) + dfs(steps - 1, idx - 1) + dfs(steps - 1, idx + 1)
            ) % (10 ** 9 + 7)

        return dfs(steps, 0)


    # 1287. Element Appearing More Than 25% In Sorted Array
    def findSpecialInteger(self, arr: List[int]) -> int:
        n = len(arr)
        for i in range(4):
            target = arr[(i + 1) * n // 4]
            l, r = 0, n
            while l < r:
                mid = l + (r - l) // 2
                if arr[mid] < target:
                    l = mid + 1
                else:
                    r = mid
            if l + n // 4 < n and arr[l + n // 4] == target:
                return arr[l]


    # 1305. All Elements in Two Binary Search Trees
    def getAllElements(self, root1: TreeNode, root2: TreeNode) -> List[int]:
        def inorder(root, ans):
            if not root:
                return
            inorder(root.left, ans)
            ans.append(root.val)
            inorder(root.right, ans)
        ans1 = []
        ans2 = []
        inorder(root1, ans1)
        inorder(root2, ans2)
        i = j = 0
        ans = []
        while i < len(ans1) and j < len(ans2):
            if ans1[i] < ans2[j]:
                ans.append(ans1[i])
                i += 1
            else:
                ans.append(ans2[j])
                j += 1
        if ans1:
            ans.extend(ans1[i:])
        if ans2:
            ans.extend(ans2[j:])
        return ans

    
    # 1314. Matrix Block Sum
    def matrixBlockSum(self, mat: List[List[int]], K: int) -> List[List[int]]:
        m, n = len(mat), len(mat[0])
        for i in range(m):
            for j in range(1, n):
                mat[i][j] += mat[i][j - 1]
        for j in range(n):
            for i in range(1, m):
                mat[i][j] += mat[i - 1][j]
        ans = [[0] * n for _ in range(m)]
        for i in range(m):
            for j in range(n):
                if i - K - 1 >= 0 and j - K - 1 >= 0:
                    ans[i][j] = (
                        mat[min(i + K, m - 1)][min(j + K, n - 1)]
                        - mat[min(i + K, m - 1)][j - K - 1]
                        - mat[i - K - 1][min(j + K, n - 1)]
                        + mat[i - K - 1][j - K - 1]
                    )
                elif i - K - 1 >= 0:
                    ans[i][j] = (
                        mat[min(i + K, m - 1)][min(j + K, n - 1)]
                        - mat[i - K - 1][min(j + K, n - 1)]
                    )
                elif j - K - 1 >= 0:
                    ans[i][j] = (
                        mat[min(i + K, m - 1)][min(j + K, n - 1)]
                        - mat[min(i + K, m - 1)][j - K - 1]
                    )
                else:
                    ans[i][j] = mat[min(i + K, m - 1)][min(j + K, n - 1)]
        return ans


    # 1329. Sort the Matrix Diagonally
    def diagonalSort(self, mat: List[List[int]]) -> List[List[int]]:
        m, n = len(mat), len(mat[0])
        def sort(i, j):
            diagonal = []
            while i < m and j < n:
                diagonal.append(mat[i][j])
                i += 1
                j += 1
            diagonal.sort()
            while i > 0 and j > 0:
                i -= 1
                j -= 1
                mat[i][j] = diagonal.pop()
        for i in range(m):
            sort(i, 0)
        for j in range(n):
            sort(0, j)
        return mat


    # 1344. Angle Between Hands of a Clock
    def angleClock(self, hour: int, minutes: int) -> float:
        diff = minutes * 6 - ((hour % 12) * 30 + minutes / 60 * 30)
        return abs(diff) if abs(diff) <= 180 else 360 - abs(diff)


    # 1356. Sort Integers by The Number of 1 Bits
    def sortByBits(self, arr: List[int]) -> List[int]:
        arr.sort(key=lambda x: (bin(x).count("1"), x))
        return arr


    # 1382. Balance a Binary Search Tree
    def balanceBST(self, root: TreeNode) -> TreeNode:
        nodes = []
        def inorder(root):
            if root:
                inorder(root.left)
                nodes.append(root)
                inorder(root.right)
        inorder(root)
        def generateBST(nodes):
            if not nodes:
                return
            mid = len(nodes) // 2
            node = nodes[mid]
            node.left = generateBST(nodes[:mid])
            node.right = generateBST(nodes[mid + 1 :])
            return node
        return generateBST(nodes)


    # 1424. Diagonal Traverse II
    def findDiagonalOrder(self, nums: List[List[int]]) -> List[int]:
        d = collections.OrderedDict()
        for i in range(len(nums)):
            for j in range(len(nums[i])):
                if i + j in d:
                    d[i + j].append(nums[i][j])
                else:
                    d[i + j] = [nums[i][j]]
        ans = []
        for k in d:
            ans += d[k][::-1]
        return ans


    # 1439. Find the Kth Smallest Sum of a Matrix With Sorted Rows
    def kthSmallest(self, mat: List[List[int]], k: int) -> int:
        ans = [0]
        for row in mat:
            ans = [-i + j for i in row for j in ans]
            heapq.heapify(ans)
            while len(ans) > k:
                heapq.heappop(ans)
        return -ans[0]


    # 1460. Make Two Arrays Equal by Reversing Sub-arrays
    def canBeEqual(self, target: List[int], arr: List[int]) -> bool:
        return collections.Counter(target) == collections.Counter(arr)


    # 1464. Maximum Product of Two Elements in an Array
    def maxProduct(self, nums: List[int]) -> int:
        first_max = second_max = float("-inf")
        for num in nums:
            if num > first_max:
                second_max = first_max
                first_max = num
            elif num > second_max:
                second_max = num
        return (first_max - 1) * (second_max - 1)


    # 1498. Number of Subsequences That Satisfy the Given Sum Condition
    def numSubseq(self, nums: List[int], target: int) -> int:
        nums.sort()
        left, right = 0, len(nums) - 1
        ans = 0
        while left <= right:
            if nums[left] + nums[right] <= target:
                ans = (ans + pow(2, right - left, 10 ** 9 + 7)) % (
                    10 ** 9 + 7
                )  # no. of combs ending at right
                left += 1
            else:
                right -= 1
        return ans


    # 1528. Shuffle String
    def restoreString(self, s: str, indices: List[int]) -> str:
        ans = [None] * len(s)
        for c, index in zip(s, indices):
            ans[index] = c
        return "".join(ans)


    # 1539. Kth Missing Positive Number
    def findKthPositive(self, arr: List[int], k: int) -> int:
        n = len(arr)
        if arr[n - 1] - n < k:
            return k + n
        l, r = 0, n - 1
        while l < r:
            mid = l + (r - l) // 2
            if arr[mid] - mid - 1 < k:
                l = mid + 1
            else:
                r = mid
        return k + l  # this is actually arr[l - 1] + k - (arr[l - 1] - l)


    # 1629. Slowest Key
    def slowestKey(self, releaseTimes: List[int], keysPressed: str) -> str:
        if len(keysPressed) <= 1:
            return keysPressed
        ans = ""
        max_duration = float("-inf")
        for i in range(len(releaseTimes)):
            if i == 0:
                duration = releaseTimes[0]
            else:
                duration = releaseTimes[i] - releaseTimes[i - 1]
            if duration > max_duration:
                max_duration = duration
                ans = keysPressed[i]
            elif duration == max_duration:
                ans = max(ans, keysPressed[i])
        return ans


    # 1630. Arithmetic Subarrays
    def checkArithmeticSubarrays(
        self, nums: List[int], l: List[int], r: List[int]
    ) -> List[bool]:
        def check(nums):
            nums = sorted(nums)
            for i in range(1, len(nums)):
                if nums[i] - nums[i - 1] != nums[1] - nums[0]:
                    return False
            return True
        res = []
        for left, right in zip(l, r):
            res.append(check(nums[left : right + 1]))
        return res


    # 1631. Path With Minimum Effort
    def minimumEffortPath(self, heights):
        m, n = len(heights), len(heights[0])
        dist = [[float("inf")] * n for _ in range(m)]
        minHeap = []
        minHeap.append((0, 0, 0))  # distance, row, col
        DIR = [0, 1, 0, -1, 0]
        while minHeap:
            d, r, c = heappop(minHeap)
            if r == m - 1 and c == n - 1:
                return d  # Reach to bottom right
            for i in range(4):
                nr, nc = r + DIR[i], c + DIR[i + 1]
                if 0 <= nr < m and 0 <= nc < n:
                    newDist = max(d, abs(heights[nr][nc] - heights[r][c]))
                    if dist[nr][nc] > newDist:
                        dist[nr][nc] = newDist
                        heappush(minHeap, (dist[nr][nc], nr, nc))


    # 1636. Sort Array by Increasing Frequency
    def frequencySort(self, nums: List[int]) -> List[int]:
        if not nums:
            return []
        counter = collections.Counter(nums)
        counter = [(val, -key) for key, val in counter.items()]
        counter.sort()
        ans = []
        for val, key in counter:
            ans.extend([-key] * val)
        return ans


    # 1637. Widest Vertical Area Between Two Points Containing No Points
    def maxWidthOfVerticalArea(self, points: List[List[int]]) -> int:
        if not points:
            return
        points.sort()
        ans = 0
        for i in range(1, len(points)):
            ans = max(ans, points[i][0] - points[i - 1][0])
        return ans


    # 1640. Check Array Formation Through Concatenation
    def canFormArray(self, arr: List[int], pieces: List[List[int]]) -> bool:
        orders = {num: i for i, num in enumerate(arr)}
        for piece in pieces:
            for i in range(1, len(piece)):
                if (
                    piece[i] in orders
                    and piece[i - 1] in orders
                    and orders[piece[i]] - orders[piece[i - 1]] == 1
                ):
                    continue
                return False
        return True


    # 1642. Furthest Building You Can Reach
    def countVowelStrings(self, n: int) -> int:
        curr = [1, 1, 1, 1, 1]
        for i in range(2, n + 1):
            next = curr[:]
            next[0] = sum(curr)
            next[1] = sum(curr[1:])
            next[2] = sum(curr[2:])
            next[3] = sum(curr[3:])
            next[4] = curr[4]
            curr = next
        return sum(curr)


    def furthestBuilding(self, heights: List[int], bricks: int, ladders: int) -> int:
        n = len(heights)
        diff_arr = [0]
        diff = 0
        for i in range(1, n):
            diff_arr.append(max(heights[i] - heights[i - 1], 0))
        i = 0
        h = []
        while i < n and len(h) < ladders:
            if diff_arr[i]:
                h.append(diff_arr[i])
            i += 1
        heapq.heapify(h)
        used_bricks = 0
        while i < n:
            if diff_arr[i]:
                if h and diff_arr[i] > h[0]:
                    used_bricks += heapq.heappop(h)
                    heapq.heappush(h, diff_arr[i])
                else:
                    used_bricks += diff_arr[i]
                if used_bricks > bricks:
                    return i - 1
            i += 1
        return n - 1 if i == n else i


    # 1646. Get Maximum in Generated Array
    def getMaximumGenerated(self, n: int) -> int:
        if n == 0:
            return 0
        if n == 1:
            return 1
        ans = [0, 1]
        for i in range(2, n + 1):
            if i % 2 == 0:
                ans.append(ans[i // 2])
            else:
                ans.append(ans[(i - 1) // 2] + ans[(i + 1) // 2])
        return max(ans)


    # 1647. Minimum Deletions to Make Character Frequencies Unique
    def minDeletions(self, s: str) -> int:
        counter = collections.Counter(collections.Counter(s).values())
        max_freq = max(counter.keys())
        arr = [0] * (max_freq + 1)
        for freq, cnt in counter.items():
            arr[freq] = cnt
        ans = 0
        for i in range(len(arr) - 1, 0, -1):
            if arr[i] > 1:
                ans += arr[i] - 1
                arr[i - 1] += arr[i] - 1
        return ans


    # 1656. Design an Ordered Stream
    class OrderedStream:
        def __init__(self, n: int):
            self.arr = [None] * n
            self.ptr = 0
        def insert(self, id: int, value: str) -> List[str]:
            self.arr[id - 1] = value
            if self.ptr == id - 1:
                for i in range(id - 1, len(self.arr)):
                    if self.arr[i]:
                        self.ptr += 1
                    else:
                        break
                return self.arr[id - 1 : self.ptr]
            return []


    # 1657. Determine if Two Strings Are Close
    def closeStrings(self, word1: str, word2: str) -> bool:
        return set(word1) == set(word2) and sorted(Counter(word1).values()) == sorted(
            Counter(word2).values()
        )
