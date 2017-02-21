import unittest

from common.text_utils.text_table import TextTable


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
            ['ONE', '%s', 5],
            ['TWO', '%s', 3],
            ['THREE', '%s', 4],
            ['FOUR', '%s', 10, TextTable.ALIGN_LEFT],
            ['FIVE'],
            ['SIX', '%s', 7, TextTable.ALIGN_CENTER],
            ['SEVEN', '%s', 4, TextTable.ALIGN_LEFT],
        ])
        print(table.header)
        self.assertEqual(table.header, "  ONE|TWO|THRE|FOUR      |FIVE|  SIX  |SEVE")

    def test_table_records(self):
        table = TextTable()
        table.add_column('ONE')
        table.add_column('TWO')
        table.add_column('THREE')

        table.add_record([1, 2, 3])
        table.add_record([4, 5, 6])

        print(table.header)
        print(table.records[0])
        print(table.records[1])

        self.assertEqual(len(table.records), 2)
        self.assertEqual(table.records[0], "  1   2     3")
        self.assertEqual(table.records[1], "  4   5     6")

if __name__ == '__main__':
    unittest.main()
