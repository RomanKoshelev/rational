import traceback


class Subscription(object):
    def __init__(self, event, method, subscriber):
        self.event = event
        self.method = method
        self.subscriber = subscriber


class EventSystem(object):
    subscrioptions = []

    @classmethod
    def send(cls, event, data) -> int:
        num = 0
        for rec in cls.subscrioptions:
            if rec.event == event:
                try:
                    if rec.subscriber is not None:
                        rec.method(rec.subscriber, data)
                    else:
                        rec.method(data)
                    num += 1
                except TypeError:
                    traceback.print_exc()
        return num

    @classmethod
    def subscribe(cls,
                  event: str,
                  method: classmethod or staticmethod,
                  subscriber: object = None
                  ) -> None:
        cls.subscrioptions.append(Subscription(event, method, subscriber))

    @classmethod
    def unsubscribe(cls,
                    event: str = None,
                    method: classmethod or staticmethod = None,
                    subscriber: object = None
                    ) -> None:
        assert event is not None or method is not None or subscriber is not None, "use unsubscribe_all"

        def same(a, b):
            return b is None or a == b

        def specified(rec):
            return same(rec.event, event) or same(rec.method, method) or same(rec.subscriber, subscriber)

        cls.subscrioptions = list(filter(specified, cls.subscrioptions))

    @classmethod
    def unsubscribe_all(cls):
        cls.subscrioptions = []
