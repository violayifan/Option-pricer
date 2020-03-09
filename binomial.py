import time
from enum import Enum
import math
import matplotlib.pyplot as plt
import numpy

class PayoffType(Enum):
    Call = 0
    Put = 1

class EuropeanOption():
    def __init__(self, expiry, strike, payoffType):
        self.expiry = expiry
        self.strike = strike
        self.payoffType = payoffType
    def payoff(self, S):
        if self.payoffType == PayoffType.Call:
            return max(S - self.strike, 0)
        elif self.payoffType == PayoffType.Put:
            return max(self.strike - S, 0)
        else:
            raise Exception("payoffType not supported: ", self.payoffType)
    def valueAtNode(self, t, S, continuation):
        if continuation == None:
            return self.payoff(S)
        else:
            return continuation

class AmericanOption():
    def __init__(self, expiry, strike, payoffType):
        self.expiry = expiry
        self.strike = strike
        self.payoffType = payoffType
    def payoff(self, S):
        if self.payoffType == PayoffType.Call:
            return max(S - self.strike, 0)
        elif self.payoffType == PayoffType.Put:
            return max(self.strike - S, 0)
        else:
            raise Exception("payoffType not supported: ", self.payoffType)
    def valueAtNode(self, t, S, continuation):
        return max(self.payoff(S), continuation)

class KnockOutOption():
    def __init__(self, downBarrier, upBarrier, barrierStart, barrierEnd, underlyingOption):
        self.underlyingOption = underlyingOption
        self.barrierStart = barrierStart
        self.barrierEnd = barrierEnd
        self.downBarrier = downBarrier
        self.upBarrier = upBarrier
        self.expiry = underlyingOption.expiry
    def payoff(self, S):
        return self.underlyingOption.payoff(S)
    def valueAtNode(self, t, S, continuation):
        if t > self.barrierStart and t < self.barrierEnd:
            if self.upBarrier != None and S > self.upBarrier:
                return 0
            elif self.downBarrier != None and S < self.downBarrier:
                return 0
        return continuation

class AsianOption():
    def __init__(self, fixings, payoffFun, As, nT):
        self.fixings = fixings
        self.payoffFun = payoffFun
        self.expiry = fixings[-1]
        self.nFixings = len(fixings)
        self.As, self.nT = nT
        self.dt = self.expiry / nT
    def onFixingDate(self, t):
        # we say t is on a fixing date if there is a fixing date T_i \in (t-dt, t]
        return filter(lambda x: x > t - self.dt and x<=t, self.fixings)
    def valueAtNode(self, t, S, continuation):
        if continuation == None:
            return [self.payoffFun((a*(self.nFixings-1) + S)/self.nFixings) for a in self.As]
        else:
            if self.onFixingDate(t):
                i = len(filter(lambda x: x < t, self.fixings)) # number of previous fixings
                Ahats = [(a*(i-1) + S)/i for a in self.As]
                nodeValues = [numpy.interp(a, self.As, continuation) for a in Ahats]
            else:
                nodeValues = continuation
        return nodeValues


# Black-Scholes analytic pricer
def cnorm(x):
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

def bsPrice(S, r, vol, T, strike, payoffType):
    fwd = S * math.exp(r * T)
    stdev = vol * math.sqrt(T)
    d1 = math.log(fwd / strike) / stdev + stdev / 2
    d2 = d1 - stdev
    if payoffType == PayoffType.Call:
        return math.exp(-r * T) * (fwd * cnorm(d1) - cnorm(d2) * strike)
    elif payoffType == PayoffType.Put:
        return math.exp(-r * T) * (strike * cnorm(-d2) - cnorm(-d1) * fwd)
    else:
        raise Exception("not supported payoff type", payoffType)

############ binomial pricer and different binomial models
def crrCalib(r, vol, t):
    b = math.exp(vol * vol * t + r * t) + math.exp(-r * t)
    u = (b + math.sqrt(b * b - 4)) / 2
    p = (math.exp(r * t) - (1 / u)) / (u - 1 / u)
    return (u, 1/u, p)

def jrrnCalib(r, vol, t):
    u = math.exp((r - vol * vol / 2) * t + vol * math.sqrt(t))
    d = math.exp((r - vol * vol / 2) * t - vol * math.sqrt(t))
    p = (math.exp(r * t) - d) / (u - d)
    return (u, d, p)

def jreqCalib(r, vol, t):
    u = math.exp((r - vol * vol / 2) * t + vol * math.sqrt(t))
    d = math.exp((r - vol * vol / 2) * t - vol * math.sqrt(t))
    return (u, d, 1/2)

def tianCalib(r, vol, t):
    v = math.exp(vol * vol * t)
    u = 0.5 * math.exp(r * t) * v * (v + 1 + math.sqrt(v*v + 2*v - 3))
    d = 0.5 * math.exp(r * t) * v * (v + 1 - math.sqrt(v*v + 2*v - 3))
    p = (math.exp(r * t) - d) / (u - d)
    return (u, d, p)

def binomialPricer(S, r, vol, trade, n, calib):
    t = trade.expiry / n
    (u, d, p) = calib(r, vol, t)
    # set up the last time slice, there are n+1 nodes at the last time slice
    vs = [trade.payoff(S * u ** (n - i) * d ** i) for i in range(n + 1)]
    # iterate backward
    for i in range(n - 1, -1, -1):
        # calculate the value of each node at time slide i, there are i nodes
        for j in range(i + 1):
            nodeS = S * u ** (i - j) * d ** j
            continuation = math.exp(-r * t) * (vs[j] * p + vs[j + 1] * (1 - p))
            vs[j] = trade.valueAtNode(t * i, nodeS, continuation)
    return vs[0]

def test1DTiming():
    opt = EuropeanOption(1, 105, PayoffType.Call)
    S, r, vol = 100, 0.01, 0.2

    bsprc = bsPrice(S, r, vol, opt.expiry, opt.strike, opt.payoffType)
    print("bsPrice = \t ", bsprc)
    n = 20
    prc = [None] * n
    timing = [None] * n
    nSteps = [None] * n
    for i in range(1, n + 1):
        nSteps[i - 1] = 20 * i
        start = time.time()
        prc[i - 1] = binomialPricer(S, r, vol, opt, nSteps[i - 1], tianCalib) - bsprc
        timing[i - 1] = time.time() - start

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(nSteps, prc, 'g')
    ax2.plot(nSteps, timing, 'b')

    ax1.set_xlabel('nTreeSteps')
    ax1.set_ylabel('Pricing Error')
    ax2.set_ylabel('Timeing')
    plt.legend()
    plt.show()

def testBTrees():
    opt = EuropeanOption(1, 105, PayoffType.Call)
    S, r, vol = 100, 0.01, 0.2

    bsprc = bsPrice(S, r, vol, opt.expiry, opt.strike, opt.payoffType)
    print("bsPrice = \t ", bsprc)
    n = 300

    crrErrs = [math.log(abs(binomialPricer(S, r, vol, opt, i, crrCalib) - bsprc)) for i in range(1, n)]
    jrrnErrs = [math.log(abs(binomialPricer(S, r, vol, opt, i, jrrnCalib) - bsprc)) for i in range(1, n)]
    jreqErrs = [math.log(abs(binomialPricer(S, r, vol, opt, i, jreqCalib) - bsprc)) for i in range(1, n)]
    tianErrs = [math.log(abs(binomialPricer(S, r, vol, opt, i, tianCalib) - bsprc)) for i in range(1, n)]

    plt.plot(range(1, n), crrErrs, label="crr")
    plt.plot(range(1, n), jrrnErrs, label="jrrn")
    plt.plot(range(1, n), jreqErrs, label="jreq")
    plt.plot(range(1, n), tianErrs, label="tian")
    plt.legend()
    plt.show()

class EuropeanPayoff():
    def __init__(self, expiry, payoffFun):
        self.expiry = expiry
        self.payoffFun = payoffFun
    def payoff(self, S):
        return self.payoffFun(S)
    def valueAtNode(self, t, S, continuation):
        return continuation

class AmericanPayoff():
    def __init__(self, expiry, payoffFun):
        self.expiry = expiry
        self.payoffFun = payoffFun
    def payoff(self, S):
        return self.payoffFun(S)
    def valueAtNode(self, t, S, continuation):
        return max(self.payoff(S), continuation)

def testAmerSpread():
    S, r, vol = 95, 0.05, 0.2
    callSpread = lambda S: min(max(S - 90, 0), 10)
    plt.plot(range(80, 120), [callSpread(i) for i in range(80, 120)])
    plt.show()
    print("Euro callspread: ", binomialPricer(S, r, vol, EuropeanPayoff(1, callSpread), 300, crrCalib))
    print("Amer callspread: ", binomialPricer(S, r, vol, AmericanPayoff(1, callSpread), 300, crrCalib))
# def testDelta():
    # crrPrc = lambda S : binomialPricer(S, r, vol, opt, n, crrCalib)
    # tianPrc = lambda S : binomialPricer(S, r, vol, opt, n, tianCalib)
    # triPrc = lambda S :  binomialPricer(S, r, vol, opt, n, tianCalib)
    # ladder = range(80, 120)
    #
    # crrDelta = [(crrPrc(s*1.001) - crrPrc(s*0.999)) / s/0.002 for s in ladder]
    # tianDelta = [(tianPrc(s * 1.001) - tianPrc(s * 0.999)) / s / 0.002 for s in ladder]
    # triDelta = [(triPrc(s * 1.001) - triPrc(s * 0.999)) / s / 0.002 for s in ladder]
    #
    # # plt.plot(ladder, crrDelta, label = "crr")
    # # plt.plot(ladder, tianDelta, label="tian")
    # plt.plot(ladder, triDelta, label="trinomial")
    # plt.legend()
    # plt.show()

if __name__ == "__main__":

    # test1DTiming()
    testAmerSpread()