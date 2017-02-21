import unittest


class TextTable(object):
    ALIGN_LEFT = -1
    ALIGN_CENTER = 0
    ALIGN_RIGHT = +1

    def __init__(self, columns=None, vline=' '):
        self._columns = []
        self._vline = vline
        if columns is not None:
            self.add_columns(columns)

    @property
    def header(self) -> str:
        return self._vline.join([self._format(c['title'], c) for c in self._columns])

    def add_column(self, title, width=None, template='%s', align=ALIGN_LEFT):
        self._columns.append({
            'title': title,
            'width': width,
            'template': template,
            'align': align,
        })

    def _format(self, val, col: dict) -> str:
        t = col['template']
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


class TextTableTests(unittest.TestCase):
    def test_table_header(self):
        table = TextTable()
        table.add_column('ONE')
        table.add_column('TWO')
        table.add_column('THREE')

        print(table.header)
        self.assertEqual(table.header, "ONE TWO THREE")

    def test_table_width(self):
        table = TextTable(vline='|')
        table.add_columns([
            ['ONE', 5],
            ['TWO', 3],
            ['THREE', 4],
            ['FOUR', 10, '%s', TextTable.ALIGN_RIGHT],
            ['FIVE'],
            ['SIX', 7, '%s', TextTable.ALIGN_CENTER],
            ['SEVEN', 4, '%s', TextTable.ALIGN_RIGHT],
        ])
        print(table.header)
        self.assertEqual(table.header, "ONE  |TWO|THRE|      FOUR|FIVE|  SIX  |SEVE")


if __name__ == '__main__':
    unittest.main()
