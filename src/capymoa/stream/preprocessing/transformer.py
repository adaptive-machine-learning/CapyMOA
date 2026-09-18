from __future__ import annotations

from abc import ABC, abstractmethod

import moa.streams.filters
from jpype import JException
from moa.streams import FilteredQueueStream

from capymoa.core import Instance

from .._stream import MOAStream, Schema


class Transformer(ABC):
    @abstractmethod
    def transform_instance(self, instance) -> Instance:
        raise NotImplementedError

    @abstractmethod
    def get_schema(self) -> Schema:
        """Return the schema of instances *leaving* this transformer.

        This is the schema the next element in a pipeline will receive, which
        is not necessarily the schema this transformer consumes -- a filter may
        add, drop or rewrite attributes. See :meth:`get_input_schema`.
        """
        raise NotImplementedError

    def get_input_schema(self) -> Schema:
        """Return the schema of instances *entering* this transformer.

        Defaults to :meth:`get_schema` for transformers that do not alter the
        attribute set.
        """
        return self.get_schema()

    @abstractmethod
    def restart(self):
        raise NotImplementedError


class MOATransformer(Transformer):
    def __init__(
        self,
        schema=None,
        moa_filter: moa.streams.filters.StreamFilter | None = None,
        CLI=None,
    ):
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
            raise RuntimeError("Must provide a moa_filter to initialize the Schema.")

        if self.schema is None:
            self.schema = self._derive_filter_schema()
            if self.schema is None:
                raise RuntimeError(
                    "Could not infer a schema from the MOA filter; pass schema= "
                    "explicitly. Filters that alter the attribute set only "
                    "publish a header once an instance has passed through them."
                )

        # Attempt to learn the post-filter schema up front. Filters that only
        # rewrite values (normalisation, added noise) publish a header
        # immediately; filters that change the attribute set publish one only
        # after the first instance, so this is retried in transform_instance.
        self._output_schema = self._derive_filter_schema()

        queue = FilteredQueueStream()
        self.filtered_stream = MOAStream(
            schema=self._output_schema or self.schema,
            moa_stream=queue,
            CLI=f"-f ({self.moa_filter.getCLICreationString(self.moa_filter.__class__)})",
        )

    def _derive_filter_schema(self) -> Schema | None:
        """Return the filter's output schema, or None if it has not published one.

        MOA filters report a null header -- or raise outright -- until they have
        enough information to describe their output, so both outcomes mean "not
        known yet" rather than an error.
        """
        try:
            moa_header = self.moa_filter.getHeader()
        except JException:
            return None
        return None if moa_header is None else Schema(moa_header=moa_header)

    @staticmethod
    def _derive_schema_from_instance(instance) -> Schema | None:
        """Return the schema a transformed instance describes itself with.

        The more reliable of the two derivations: some filters (the hashing
        trick, for one) never publish a header on the filter object, but every
        instance they emit still carries the dataset header it belongs to.
        """
        try:
            dataset = instance.java_instance.instance.dataset()
        except (JException, AttributeError):
            return None
        return None if dataset is None else Schema(moa_header=dataset)

    def __str__(self):
        moa_filter_str = str(
            self.moa_filter.getCLICreationString(self.moa_filter.__class__)
        )
        moa_filter_str = moa_filter_str.removesuffix(" ")
        return f"Transformer({moa_filter_str})"

    def transform_instance(self, instance) -> Instance:
        # MOA filters are not stateless.
        # This hack avoids transforming an instance twice.
        if self._last_instance == instance:
            return self._last_transformed_instance
        self._last_instance = instance

        self.filtered_stream.moa_stream.addToQueue(instance.java_instance.instance)
        new_instance = self.filtered_stream.next_instance()

        if self._output_schema is None:
            # A filter that changes the attribute set only describes its output
            # once it has seen an instance. Prefer the header the instance
            # itself carries -- filters that never publish one on the filter
            # object still stamp it on what they emit.
            self._output_schema = (
                self._derive_schema_from_instance(new_instance)
                or self._derive_filter_schema()
            )
            if self._output_schema is not None:
                self.filtered_stream.schema = self._output_schema

        self._last_transformed_instance = new_instance
        return new_instance

    def get_schema(self) -> Schema:
        """Return the schema of instances leaving this transformer.

        Falls back to the input schema while the underlying MOA filter has not
        published a header yet.
        """
        return self._output_schema or self.schema

    def get_input_schema(self) -> Schema:
        """Return the schema of instances entering this transformer."""
        return self.schema

    def restart(self):
        self.moa_filter.restart()

    def get_moa_filter(self):
        return self.moa_filter
