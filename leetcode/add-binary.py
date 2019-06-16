class Solution(object):
    def addBinary(self, a, b):
        """
        :type a: str
        :type b: str
        :rtype: str
        """
        res = []
        carry, val = 0, 0
        if len(a) < len(b):
            a, b = b, a
        for i in xrange(len(a)):
            val = carry
            val += int(a[len(a) - i -1])
            if i < len(b):
                val += int(b[len(b) - i - 1])
            val, carry = val % 2, val / 2
            res.append(str(val))
        if carry:
            res.append(str(carry))
        return "".join(res[::-1])
