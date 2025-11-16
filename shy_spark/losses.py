class losses:
    def __init__(self, epsilon, neta, w):
        self.epsilon = epsilon
        self.neta = neta
        self.w = w
        pass

    def _Lpred(self):
        return

    def _Lfidleity(self):
        return

    def _Ldistinct(self):
        return

    def _Lalpha(self):
        return
    
    def loss(self):
        L = self._Lpred + self.epsilon*self._Lfidleity+self.neta*self._Ldistinct+self.w*self._Lalpha
        return L