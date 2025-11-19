class losses:
    def __init__(self, epsilon, neta, w):
        self.epsilon = epsilon
        self.neta = neta
        self.w = w
        pass

    def Lpred(self):
        return

    def Lfidleity(self):
        return

    def Ldistinct(self):
        return

    def Lalpha(self):
        return
    
    def loss(self):
        L = self.Lpred + self.epsilon*self.Lfidleity+self.neta*self.Ldistinct+self.w*self.Lalpha
        return L