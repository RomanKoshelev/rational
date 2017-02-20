import unittest

from common.events import Events


class Subscriber(object):
    def __init__(self):
        Events.subscribe('test_event', Subscriber.on_event, self)

    def on_event(self, data):
        print(self, data)


class EventTests(unittest.TestCase):
    def test_object_subscription(self):
        Events.unsubscribe_all()
        _ = Subscriber()
        num = Events.send('test_event', ([1, 2], "3"))
        self.assertEqual(num, 1)

    def test_lambda_subscription(self):
        Events.unsubscribe_all()
        Events.subscribe('test_event', lambda data: print(data))
        num = Events.send('test_event', ([1, 2], "3"))
        self.assertEqual(num, 1)


if __name__ == '__main__':
    unittest.main()
