class Solution(object):
    def getHint(self, secret, guess):
        """
        :type secret: str
        :type guess: str
        :rtype: str
        """
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
        return str(bull) + 'A' + str(cow) + 'B'
