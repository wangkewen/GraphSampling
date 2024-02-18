class MetricsProcessor(object):
    r"""
    Process and manipulate metrics data
    Args: None
    """

    def __init__(self, opt='sum', ispercent=True, isacc=True):
        self.opt = opt
        self.ispercent = ispercent
        self.isacc = isacc

    def intervalCompute(self, ymap, x1, x2):
        """
        bin stat(avg,sum,count,...) of ymap(x1) in interval points of x2 
        """
        operation = self.isacc
        y1 = list()
        prex = 0
        acc = 0
        count = 0
        for e in x1:
            if e >= x2[prex]:
                value = 0.0
                if count > 0:
                    if operation == 'avg':
                        value = (acc * 1.0) / count;
                    elif operation == 'count':
                        value = count
                    elif operation == 'sum':
                        value = acc * 1.0;
                    else:
                        value = acc
                y1.append(value)
                count = 0
                acc = 0
                prex += 1
                if prex == len(x2):
                    break
            acc += ymap[e]
            count += 1
        
        if count > 0:
            if operation == 'avg':
                value = (acc * 1.0) / count;
            elif operation == 'count':
                value = count
            elif operation == 'sum':
                value = acc * 1.0;
            else:
                value = acc
            y1.append(value)

        # use zero for empty values
        while len(y1) < len(x2):
            y1.append(0.0)
        return y1

    def findSampleInterval(self, x):
        """
        select points with interval=gap
        """
        xis = list()
        best = 20
        limit = 200
        gap = 1
        lenx = len(x)
        if lenx > limit:
            gap = 10
        elif lenx > best:
            gap = lenx // best
        for e in range(len(x)):
            if e % gap == 0:
                xis.append(x[e])
        return xis

    def accumulative(self, ls):
        """
        y = sum(x<X), F
        """
        return [sum(ls[0:x:1]) for x in range(1, len(ls)+1)]

    def generateY(self, xlist, xis, ymap=None):
        """
        bin stat x,y by interval xis
        """
        x1 = sorted(list(xlist))
        if not ymap:
            ymap = {e:e for e in x1}
        y1 = self.intervalCompute(ymap, x1, xis)
        if self.ispercent:
            y1 = [e*1.0/(sum(y1) if sum(y1) != 0 else 1) for e in y1]
        if self.isacc:
            y1 = self.accumulative(y1)
        return x1, y1

    def d_stat(self, lxy):
        """
        D-statistics
        Args: lxy list[x, y, y1, y2, y3,...]
        """
        yds = list()
        for i in range(2, len(lxy)):
            ymax = 0.5
            if not lxy[i]:
                yds.append(ymax)
                continue
            ymax = 0
            for j in range(len(lxy[1])):
                ymax = max(ymax, abs(lxy[1][j] - lxy[i][j]))
            yds.append(ymax)
        return yds
