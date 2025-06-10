class CrossWalk:
    def __init__(self):
        self._id = None
        self._name = None
        self._sPos = None
        self._tPos = None
        self._hdg = None
        self._corners = []

    @property
    def sPos(self):
        return self._sPos

    @sPos.setter
    def sPos(self, value):
        self._sPos = float(value)

    @property
    def tPos(self):
        return self._tPos

    @tPos.setter
    def tPos(self, value):
        self._tPos = float(value)

    @property
    def hdg(self):
        return self._hdg

    @hdg.setter
    def hdg(self, value):
        self._hdg = float(value)

    @property
    def id(self):
        return self._id

    @id.setter
    def id(self, value):
        self._id = int(value)

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    @property
    def type(self):
        return self._type

    @type.setter
    def type(self, value):
        self._type = value

    @property
    def corners(self):
        return self._corners

    @corners.setter
    def corners(self, value):
        self._corners = value
