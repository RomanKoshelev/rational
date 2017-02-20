import traceback


class Record(object):
    def __init__(self, name, method, subscriber):
        self.name = name
        self.method = method
        self.subscriber = subscriber

    def __str__(self):
        return "name: %s, method: %s, subscriber: %s" % (self.name, self.subscriber, self.method)


class Events(object):
    records = set()

    @classmethod
    def send(cls, name, data) -> int:
        num = 0
        for rec in cls.records:
            if rec.name == name:
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
    def subscribe(cls, name, method, subscriber=None):
        cls.records.add(Record(name, method, subscriber))

    @classmethod
    def unsubscribe_all(cls):
        cls.records = set()

