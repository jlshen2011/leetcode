# 146. LRU Cache
class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.dict = collections.OrderedDict()

    def get(self, key: int) -> int:
        if key in self.dict:
            value = self.dict.pop(key=key)
            self.dict[key] = value
            return value
        else:
            return -1

    def put(self, key: int, value: int) -> None:
        if key in self.dict:
            self.dict.pop(key=key)
            self.dict[key] = value
        elif len(self.dict) < self.capacity:
            self.dict[key] = value
        else:
            self.dict.popitem(last=False)
            self.dict[key] = value


# 155. Min Stack
class MinStack:
    def __init__(self):
        self.list = []

    def push(self, x: int) -> None:
        if not self.list or x < self.list[-1][1]:
            self.list.append([x, x])
        else:
            self.list.append([x, self.list[-1][1]])

    def pop(self) -> None:
        self.list.pop()

    def top(self) -> int:
        if self.list:
            return self.list[-1][0]

    def getMin(self) -> int:
        if self.list:
            return self.list[-1][1]


# 173. Binary Search Tree Iterator
class BSTIterator:
    def __init__(self, root: TreeNode):
        self.root = root
        self.stack = []
        self.leftmost(root)

    def leftmost(self, node):
        while node:
            self.stack.append(node)
            node = node.left

    def next(self) -> int:
        if self.hasNext():
            node = self.stack.pop()
            if node:
                self.leftmost(node.right)
            return node.val

    def hasNext(self) -> bool:
        return len(self.stack) > 0


# 208. Implement Trie (Prefix Tree)
class Trie:
    def __init__(self):
        self.trie = {}

    def insert(self, word: str) -> None:
        node = self.trie
        for ch in word:
            if ch not in node:
                node[ch] = {}
            node = node[ch]
        node["$"] = True

    def search(self, word: str) -> bool:
        node = self.trie
        for ch in word:
            if ch not in node:
                return False
            node = node[ch]
        return "$" in node

    def startsWith(self, prefix: str) -> bool:
        node = self.trie
        for ch in prefix:
            if ch not in node:
                return False
            node = node[ch]
        return True


# 211. Design Add and Search Words Data Structure
class WordDictionary:
    def __init__(self):
        self.trie = {}

    def addWord(self, word: str) -> None:
        node = self.trie
        for ch in word:
            if ch not in node:
                node[ch] = {}
            node = node[ch]
        node["$"] = True

    def search(self, word: str) -> bool:
        def helper(word, node):
            for i, ch in enumerate(word):
                if ch in node:
                    node = node[ch]
                else:
                    if ch == ".":
                        ans = False
                        for next_ch in node:
                            if next_ch != "$":
                                ans = ans or helper(word[i + 1 :], node[next_ch])
                        return ans
                    else:
                        return False
            return "$" in node
        return helper(word, self.trie)



# 225. Implement Stack using Queues
class Stack(object):
    def __init__(self):
        self.data = collections.deque()

    def push(self, x):
        self.data.append(x)
        for i in xrange(len(self.data) - 1):
            self.data.append(self.data.popleft())

    def pop(self):
        self.data.popleft()

    def top(self):
        return self.data[0]

    def empty(self):
        return len(self.data) == 0


# 232. Implement Queue using Stacks
class Queue(object):
    def __init__(self):
        self.A, self.B = [], []

    def push(self, x):
        self.A.append(x)

    def pop(self):
        self.peek()
        self.B.pop()

    def peek(self):
        if not self.B:
            while self.A:
                self.B.append(self.A.pop())
        return self.B[-1]

    def empty(self):
        return not self.A and not self.B


# 284. Peeking Iterator
class PeekingIterator:
    def __init__(self, iterator):
        self.iterator = iterator
        self.cache = None

    def peek(self):
        if self.cache:
            return self.cache
        elif self.iterator.hasNext():
            self.cache = self.iterator.next()
            return self.cache

    def next(self):
        if self.cache:
            tmp = self.cache
            self.cache = None
            return tmp
        else:
            return self.iterator.next()

    def hasNext(self):
        """
        :rtype: bool
        """
        if self.cache:
            return True
        else:
            return self.iterator.hasNext()


# 295. Find Median from Data Stream
class MedianFinder:
    def __init__(self):
        self.min_h = []
        self.max_h = []

    def addNum(self, num: int) -> None:
        if not self.min_h or num < self.min_h[0]:
            heapq.heappush(self.max_h, -num)
        else:
            heapq.heappush(self.min_h, num)
        if len(self.max_h) > len(self.min_h) + 1:
            top = -heapq.heappop(self.max_h)
            heapq.heappush(self.min_h, top)
        elif len(self.max_h) < len(self.min_h) - 1:
            top = heapq.heappop(self.min_h)
            heapq.heappush(self.max_h, -top)

    def findMedian(self) -> float:
        if len(self.min_h) == len(self.max_h):
            return (self.min_h[0] - self.max_h[0]) / 2
        if len(self.min_h) == len(self.max_h) + 1:
            return self.min_h[0]
        if len(self.min_h) + 1 == len(self.max_h):
            return -self.max_h[0]


# 297. Serialize and Deserialize Binary Tree
class Codec:
    def serialize(self, root):
        path = []
        def helper(root):
            if not root:
                path.append("Null")
                return
            path.append(str(root.val))
            helper(root.left)
            helper(root.right)
        helper(root)
        return ",".join(path)

    def deserialize(self, data):
        def helper(data):
            if not data:
                return
            if data[-1] == "Null":
                data.pop()
                return
            root = TreeNode(int(data[-1]))
            data.pop()
            root.left = helper(data)
            root.right = helper(data)
            return root
        data = data.split(",")[::-1]
        return helper(data)


# 303. Range Sum Query - Immutable
class NumArray:
    def __init__(self, nums: List[int]):
        self.sums = []
        sums = 0
        for num in nums:
            sums += num
            self.sums.append(sums)
    def sumRange(self, i: int, j: int) -> int:
        return self.sums[j] - (self.sums[i - 1] if i > 0 else 0)


# 304. Range Sum Query 2D - Immutable
class NumMatrix:
    def __init__(self, matrix: List[List[int]]):
        if len(matrix) == 0:
            return
        self.dp = matrix
        self.m = len(matrix)
        self.n = len(matrix[0])
        for i in range(1, self.m):
            for j in range(0, self.n):
                self.dp[i][j] += self.dp[i - 1][j]
        for i in range(0, self.m):
            for j in range(1, self.n):
                self.dp[i][j] += self.dp[i][j - 1]

    def sumRegion(self, row1: int, col1: int, row2: int, col2: int) -> int:
        if row1 == 0 and col1 == 0:
            return self.dp[row2][col2]
        if row1 > 0 and col1 == 0:
            return self.dp[row2][col2] - self.dp[row1 - 1][col2]
        if row1 == 0 and col1 > 0:
            return self.dp[row2][col2] - self.dp[row2][col1 - 1]
        else:
            return (
                self.dp[row2][col2]
                - self.dp[row1 - 1][col2]
                - self.dp[row2][col1 - 1]
                + self.dp[row1 - 1][col1 - 1]
            )

    def maxProfit(self, prices: List[int]) -> int:
        sold, held, reset = float("-inf"), float("-inf"), 0
        for price in prices:
            pre_sold = sold
            sold = held + price
            held = max(held, reset - price)
            reset = max(reset, pre_sold)
        return max(reset, sold)

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


# 341. Flatten Nested List Iterator
class NestedIterator:
    def __init__(self, nestedList: [NestedInteger]):
        self.list = []
        def flatten(nestedList):
            for item in nestedList:
                if item.isInteger():
                    self.list.append(item.getInteger())
                else:
                    flatten(item.getList())
        flatten(nestedList)
        self.list = self.list[::-1]

    def next(self) -> int:
        if self.list:
            return self.list.pop()

    def hasNext(self) -> bool:
        return len(self.list) > 0

    def isPowerOfFour(self, num):
        if num <= 0:
            return False
        while num % 4 == 0:
            num /= 4
        return num == 1

    def reverseString(self, s: List[str]) -> None:
        i, j = 0, len(s) - 1
        while i < j:
            s[i], s[j] = s[j], s[i]
            i += 1
            j -= 1

    def reverseVowels(self, s):
        l = list(s)
        i, j = 0, len(l) - 1
        while i < j:
            while i < j and l[i] not in "aeiouAEIOU":
                i += 1
            while i < j and l[j] not in "aeiouAEIOU":
                j -= 1
            if i < j:
                l[i], l[j] = l[j], l[i]
                i += 1
                j -= 1
        return "".join(l)


# 380. Insert Delete GetRandom O(1)
class RandomizedSet:
    def __init__(self):
        self.dict = dict()
        self.list = []

    def insert(self, val: int) -> bool:
        if val in self.dict:
            return False
        self.dict[val] = len(self.list)
        self.list.append(val)
        return True

    def remove(self, val: int) -> bool:
        if val not in self.dict:
            return False
        self.list[self.dict[val]] = self.list[-1]
        self.dict[self.list[-1]] = self.dict[val]
        self.list.pop()
        del self.dict[val]
        return True

    def getRandom(self) -> int:
        return self.list[random.randint(0, len(self.list) - 1)]


# 381. Insert Delete GetRandom O(1) - Duplicates allowed
class RandomizedCollection:
    def __init__(self):
        self.map = collections.defaultdict(set)
        self.list = []

    def insert(self, val: int) -> bool:
        flag = True if val not in self.map else False
        self.map[val].add(len(self.list))
        self.list.append(val)
        return flag

    def remove(self, val: int) -> bool:
        if val not in self.map:
            return False
        idx = self.map[val].pop()
        if idx < len(self.list) - 1:
            self.map[self.list[-1]].remove(len(self.list) - 1)
            self.map[self.list[-1]].add(idx)
            self.list[-1], self.list[idx] = self.list[idx], self.list[-1]
        self.list.pop()
        if not self.map[val]:
            del self.map[val]
        return True

    def getRandom(self) -> int:
        return self.list[random.randint(0, len(self.list) - 1)]


# 398. Random Pick Index
class RandomPickIndex:
    def __init__(self, nums: List[int]):
        self.nums = nums

    def pick(self, target: int) -> int:
        count = 0
        for i, num in enumerate(self.nums):
            if num == target:
                count += 1
                if random.randint(1, count) == count:
                    res = i
        return res


# 432. All O`one Data Structure
class AllOne(object):
    def __init__(self):
        self.head = Node()
        self.tail = Node()
        self.head.next = self.tail
        self.tail.prev = self.head
        self.cache = collections.defaultdict(Node)

    def inc(self, key):
        if not key in self.cache:
            cur = self.head
        else:
            cur = self.cache[key]
            cur.key_set.remove(key)

        if cur.num + 1 != cur.next.num:
            new_node = Node(cur.num + 1)
            self._insert_after(cur, new_node)
        else:
            new_node = cur.next
        new_node.key_set.add(key)
        self.cache[key] = new_node
        if not cur.key_set and cur.num != 0:
            self._remove(cur)

    def dec(self, key):
        if not key in self.cache:
            return

        cur = self.cache[key]
        self.cache.pop(key)
        cur.key_set.remove(key)

        if cur.num != 1:
            if cur.num - 1 != cur.prev.num:
                new_node = Node(cur.num - 1)
                self._insert_after(cur.prev, new_node)
            else:
                new_node = cur.prev
            new_node.key_set.add(key)
            self.cache[key] = new_node

        if not cur.key_set and cur.num != 0:
            self._remove(cur)

    def getMaxKey(self):
        if self.tail.prev.num == 0:
            return ""
        key = (
            self.tail.prev.key_set.pop()
        )  # pop and add back to get arbitrary (but not random) element
        self.tail.prev.key_set.add(key)
        return key

    def getMinKey(self):
        if self.head.next.num == 0:
            return ""
        key = self.head.next.key_set.pop()
        self.head.next.key_set.add(key)
        return key

    def _remove(self, node):
        p = node.prev
        n = node.next
        p.next = n
        n.prev = p

    def _insert_after(self, node, new_block):
        old_after = node.next
        node.next = new_block
        new_block.prev = node
        new_block.next = old_after
        old_after.prev = new_block


# 449. Serialize and Deserialize BST
class Codec:
    def serialize(self, root: TreeNode) -> str:
        def helper(root):
            if not root:
                return
            ans.append(str(root.val))
            helper(root.left)
            helper(root.right)
        ans = []
        helper(root)
        return ",".join(ans)

    def deserialize(self, data: str) -> TreeNode:
        def helper(data, lo, hi):
            if data and lo < int(data[-1]) < hi:
                val = int(data.pop())
                root = TreeNode(val)
                root.left = helper(data, lo, val)
                root.right = helper(data, val, hi)
                return root
        if data == "":
            return
        data = data.split(",")[::-1]
        return helper(data, float("-inf"), float("inf"))


# 528. Random Pick with Weight
class RandomPickWithWeight:
    def __init__(self, w: List[int]):
        self.w = []
        sum = 0
        for weight in w:
            sum += weight
            self.w.append(sum)

    def pickIndex(self) -> int:
        w = random.random() * self.w[-1]
        l, r = 0, len(self.w)
        while l < r:
            mid = l + (r - l) // 2
            if self.w[mid] < w:
                l = mid + 1
            else:
                r = mid
        return l


# 622. Design Circular Queue
class MyCircularQueue:
    def __init__(self, k: int):
        self.queue = [0] * k
        self.head_idx = 0
        self.count = 0
        self.k = k

    def enQueue(self, value: int) -> bool:
        if self.count == self.k:
            return False
        self.queue[(self.head_idx + self.count) % self.k] = value
        self.count += 1
        return True

    def deQueue(self) -> bool:
        if self.count == 0:
            return False
        self.head_idx = (self.head_idx + 1) % self.k
        self.count -= 1
        return True

    def Front(self) -> int:
        if self.count == 0:
            return -1
        return self.queue[self.head_idx]

    def Rear(self) -> int:
        if self.count == 0:
            return -1
        return self.queue[(self.head_idx + self.count - 1) % self.k]

    def isEmpty(self) -> bool:
        return self.count == 0

    def isFull(self) -> bool:
        return self.count == self.k


# 703. Kth Largest Element in a Stream
class KthLargest:
    def __init__(self, k: int, nums: List[int]):
        heapq.heapify(nums)
        while len(nums) > k:
            heapq.heappop(nums)
        self.h = nums
        self.k = k

    def add(self, val: int) -> int:
        if len(self.h) < self.k:
            heapq.heappush(self.h, val)
        elif val > self.h[0]:
            heapq.heappop(self.h)
            heapq.heappush(self.h, val)
        return self.h[0]


# 706. Design HashMap
class Bucket:
    def __init__(self):
        self.bucket = []

    def put(self, key, value):
        for i, (k, v) in enumerate(self.bucket):
            if k == key:
                self.bucket[i] = (key, value)
                return
        self.bucket.append((key, value))

    def get(self, key):
        for k, v in self.bucket:
            if k == key:
                return v
        return -1

    def remove(self, key):
        for i, (k, v) in enumerate(self.bucket):
            if k == key:
                del self.bucket[i]

class MyHashMap:
    def __init__(self):
        self.key_space = 412
        self.hash_table = [Bucket() for _ in range(self.key_space)]

    def put(self, key: int, value: int) -> None:
        hash_key = key % self.key_space
        return self.hash_table[hash_key].put(key, value)

    def get(self, key: int) -> int:
        hash_key = key % self.key_space
        return self.hash_table[hash_key].get(key)

    def remove(self, key: int) -> None:
        hash_key = key % self.key_space
        return self.hash_table[hash_key].remove(key)



# 919. Complete Binary Tree Inserter
class CBTInserter:
    def __init__(self, root: TreeNode):
        self.root = root
        self.deque = collections.deque()
        q = collections.deque([root])
        while q:
            node = q.popleft()
            if not node.left or not node.right:
                self.deque.append(node)
            if node.left:
                q.append(node.left)
            if node.right:
                q.append(node.right)

    def insert(self, v: int) -> int:
        self.deque.append(TreeNode(v))
        node = self.deque[0]
        if not node.left:
            node.left = self.deque[-1]
        else:
            node.right = self.deque[-1]
            self.deque.popleft()
        return node.val

    def get_root(self) -> TreeNode:
        return self.root


# 1032. Stream of Characters
class StreamChecker:
    def __init__(self, words: List[str]):
        self.trie = {}
        self.stream = deque()
        for word in words:
            node = self.trie
            for ch in word[::-1]:
                if ch not in node:
                    node[ch] = {}
                node = node[ch]
            node["$"] = True

    def query(self, letter: str) -> bool:
        self.stream.appendleft(letter)
        node = self.trie
        for ch in self.stream:
            if "$" in node:
                return True
            if ch not in node:
                return False
            node = node[ch]
        return "$" in node


# 1146. Snapshot Array
class SnapshotArray:
    def __init__(self, length: int):
        self.dict = [{} for _ in range(length)]
        self.cur_ver = 0

    def set(self, index: int, val: int) -> None:
        self.dict[index][self.cur_ver] = val

    def snap(self) -> int:
        self.cur_ver += 1
        return self.cur_ver - 1

    def get(self, index: int, snap_id: int) -> int:
        if snap_id in self.dict[index]:
            return self.dict[index][snap_id]
        last = 0
        for i in range(self.cur_ver + 1):
            if i in self.dict[index]:
                last = self.dict[index][i]
            if i >= snap_id:
                return last
        return last


# 1352. Product of the Last K Numbers
class ProductOfNumbers(object):
    def __init__(self):
        self.curlist = []
        self.product = 1
        
    def add(self, num):
        if num == 0:
            self.product = 1 
            self.curlist = []
        else:
            self.product = self.product *num
            self.curlist.append(self.product)

    def getProduct(self, k):
        if k == len(self.curlist):
            return self.product
        elif k > len(self.curlist):
            return 0 
        else:
            return int(self.curlist[-1]/self.curlist[-k - 1])


# 1352. Product of the Last K Numbers
class ProductOfNumbers(object):
    def __init__(self):
        self.curlist = []
        self.product = 1

    def add(self, num):
        if num == 0:
            self.product = 1
            self.curlist = []
        else:
            self.product = self.product * num
            self.curlist.append(self.product)

    def getProduct(self, k):
        if k == len(self.curlist):
            return self.product
        elif k > len(self.curlist):
            return 0
        else:
            return int(self.curlist[-1] / self.curlist[-k - 1])


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