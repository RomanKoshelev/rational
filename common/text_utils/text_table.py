class TextTable(object):
    ALIGN_LEFT = -1
    ALIGN_CENTER = 0
    ALIGN_RIGHT = +1

    def __init__(self, columns=None, vline=' '):
        self._columns = []
        self.records = []
        self._vline = vline
        if columns is not None:
            self.add_columns(columns)

    @property
    def header(self) -> str:
        return self._vline.join([self._format(c['title'], c, header=True) for c in self._columns])

    def add_column(self, title, width=None, template='%s', align=ALIGN_RIGHT):
        self._columns.append({
            'title': title,
            'width': width if width is not None else len(title),
            'template': template,
            'align': align,
        })

    def _format(self, val, col: dict, header=False) -> str:
        t = '%s' if header else col['template']
        w = col['width']
        a = col['align']
        s = str(t % val)[0:w]
        l = len(s)
        if w is not None and l < w:
            d = (w - l)
            if a == self.ALIGN_LEFT:
                s += ' ' * d
            if a == self.ALIGN_RIGHT:
                s = ' ' * d + s
            if a == self.ALIGN_CENTER:
                s = ' ' * (d // 2) + s + ' ' * (d - d // 2)
        return s

    def add_columns(self, columns: list):
        for col in columns:
            if len(col) == 1:
                assert type(col[0]) is str, "title:%s" % col[0]
                col.append(None)
            if len(col) == 2:
                assert type(col[1]) is int or col[1] is None, "title:%s, width:%s" % (col[0], col[1])
                col.append('%s')
            if len(col) == 3:
                assert type(col[2]) is str, "title:%s, template:%s" % (col[0], col[2])
                col.append(self.ALIGN_LEFT)
            self.add_column(col[0], col[1], col[2], col[3])

    def add_record(self, vals):
        self.records.append(self._vline.join(
            [self._format(c[0], c[1]) for c in zip(vals, self._columns)]
        ))