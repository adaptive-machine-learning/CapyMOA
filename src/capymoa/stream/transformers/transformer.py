from capymoa.stream import Stream


class FilteredStream(Stream):
    def __init__(self, schema=None, CLI=None, moa_filter=None):
        super(FilteredStream, self).__init__(schema, CLI, moa_filter)