import unittest

from common.events import EventSystem, Subscriber


class TestSubscriber(Subscriber):
    def __init__(self):
        self._subscribe('test_event', TestSubscriber.on_event)

    def on_event(self, data):
        print(self, data)


class EventTests(unittest.TestCase):
    def test_object_subscription(self):
        EventSystem.unsubscribe_all()
        _ = TestSubscriber()
        num = EventSystem.send('test_event', ([1, 2], "3"))
        self.assertEqual(num, 1)

    def test_lambda_subscription(self):
        EventSystem.unsubscribe_all()
        EventSystem.subscribe('test_event', lambda data: print(data))
        num = EventSystem.send('test_event', ([1, 2], "3"))
        self.assertEqual(num, 1)


if __name__ == '__main__':
    unittest.main()
