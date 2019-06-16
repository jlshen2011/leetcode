class Solution:
    # @param a, a string
    # @param b, a string
    # @return a boolean
    def compareVersion(self, version1, version2):
        v1 = version1.split(".")
        v2 = version2.split(".")
        len1 = len(v1)
        len2 = len(v2)
        for i in xrange(max(len1, len2)):
            v1Token = 0
            if i < len1:
                v1Token = int(v1[i])
            v2Token = 0
            if i < len2:
                v2Token = int(v2[i])
            if v1Token < v2Token:
                return -1
            if v1Token > v2Token:
                return 1
        return 0
