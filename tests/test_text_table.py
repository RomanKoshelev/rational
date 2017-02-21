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
            ['ONE', 5],
            ['TWO', 3],
            ['THREE', 4],
            ['FOUR', 10, '%s', TextTable.ALIGN_RIGHT],
            ['FIVE'],
            ['SIX', 7, '%s', TextTable.ALIGN_CENTER],
            ['SEVEN', 4, '%s', TextTable.ALIGN_LEFT],
        ])
        print(table.header)
        self.assertEqual(table.header, "ONE  |TWO|THRE|      FOUR|FIVE|  SIX  |SEVE")

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
