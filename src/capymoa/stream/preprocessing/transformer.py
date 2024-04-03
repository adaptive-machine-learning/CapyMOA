from __future__ import annotations

from abc import ABC, abstractmethod

from capymoa.stream import Schema, Stream
from capymoa.stream.instance import Instance
from moa.streams import FilteredQueueStream
from moa.streams.filters import StreamFilter


class Transformer(ABC):
    @abstractmethod
    def transform_instance(self, instance) -> Instance:
        raise NotImplementedError

    @abstractmethod
    def get_schema(self) -> Schema:
        raise NotImplementedError

    @abstractmethod
    def restart(self):
        raise NotImplementedError


class MOATransformer(Transformer):
    def __init__(self, schema=None, moa_filter: StreamFilter | None = None, CLI=None):
        self.schema = schema
        self.CLI = CLI
        self.moa_filter = moa_filter
        self._last_instance = None
        self._last_transformed_instance = None

        if self.CLI is not None:
            if self.moa_filter is not None:
                self.moa_filter.getOptions().setViaCLIString(CLI)
            else:
                raise RuntimeError("Must provide a moa_filter to set via CLI.")

        if self.moa_filter is not None:
            # Must call this method exactly here, because prepareForUse invoke the method to initialize the
            # header file of the stream (synthetic ones)
            self.moa_filter.prepareForUse()
        else:
            raise RuntimeError(
                "Must provide a moa_filter to initialize the Schema."
            )

        if self.schema is None:
            if self.moa_filter is not None:
                self.schema = Schema(moa_header=self.moa_filter.getHeader())
            else:
                raise RuntimeError(
                    "Must provide a moa_filter to initialize the Schema."
                )

        self.filtered_stream = Stream(schema=schema,
                                      moa_stream=FilteredQueueStream(),
                                      CLI=f"-f ({self.moa_filter.getCLICreationString(self.moa_filter.__class__)})")

    def __str__(self):
        return f"Transformer({str(self.moa_filter.getCLICreationString(self.moa_filter.__class__))})"

    def transform_instance(self, instance) -> Instance:
        # MOA filters are not stateless.
        # This hack avoids transforming an instance twice.
        if self._last_instance == instance:
            return self._last_transformed_instance
        self._last_instance = instance

        self.filtered_stream.moa_stream.addToQueue(instance.java_instance.instance)
        new_instance = self.filtered_stream.next_instance()

        self._last_transformed_instance = new_instance
        return new_instance

    def get_schema(self):
        return self.schema

    def restart(self):
        self.moa_filter.restart()

    def get_moa_filter(self):
        return self.moa_filter



