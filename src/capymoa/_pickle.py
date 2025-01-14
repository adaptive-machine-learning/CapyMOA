"""This file is a hacky workaround for https://github.com/jpype-project/jpype/issues/1201
TODO: When the issue is resolved, remove this file and update the required
version of JPype to the one that includes the fix.

This is a patched version of https://github.com/jpype-project/jpype/blob/653ccffd1df46e4d472217d77f592326ae3d3690/jpype/pickle.py
"""

# *****************************************************************************
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
#   See NOTICE file for details.
#
# *****************************************************************************

import _jpype
import pickle
from copyreg import dispatch_table

__ALL__ = ["JPickler", "JUnpickler"]

# This must exist as a global, the real unserializer is created by the JUnpickler


def encode(object_):
    from java.io import ObjectOutputStream, ByteArrayOutputStream

    o_stream = ByteArrayOutputStream()
    oo_stream = ObjectOutputStream(o_stream)
    oo_stream.writeObject(object_)
    return o_stream.toByteArray()


class JUnserializer(object):
    def __call__(self, *args):
        raise pickle.UnpicklingError("Unpickling Java requires JUnpickler")


class _JDispatch(object):
    """Dispatch for Java classes and objects.

    Python does not have a good way to register a reducer that applies to
    many classes, thus we will substitute the usual dictionary with a
    class that can produce reducers as needed.
    """

    def __init__(self, dispatch):
        self._builder = JUnserializer()
        self._dispatch = dispatch

        # Extension dispatch table holds reduce method
        self._call = self.reduce

    # Pure Python _Pickler uses get()
    def get(self, cls):
        if not issubclass(cls, (_jpype.JClass, _jpype.JObject)):
            return self._dispatch.get(cls)
        return self._call

    # Python3 cPickler uses __getitem__()
    def __getitem__(self, cls):
        if not issubclass(cls, (_jpype.JClass, _jpype.JObject)):
            return self._dispatch[cls]
        return self._call

    def reduce(self, obj):
        byte = bytes(encode(obj))
        return (self._builder, (byte,))


class JPickler(pickle.Pickler):
    """Pickler overloaded to support Java objects

    Parameters:
        file: a file or other writeable object.
        *args: any arguments support by the native pickler.

    Raises:
        java.io.NotSerializableException: if a class is not serializable or
            one of its members
        java.io.InvalidClassException: an error occures in constructing a
            serialization.

    """

    def __init__(self, file, *args, **kwargs):
        pickle.Pickler.__init__(self, file, *args, **kwargs)

        # In Python3 we need to hook into the dispatch table for extensions
        self.dispatch_table = _JDispatch(dispatch_table)


class JUnpickler(pickle.Unpickler):
    """Unpickler overloaded to support Java objects

    Parameters:
        file: a file or other readable object.
        *args: any arguments support by the native unpickler.

    Raises:
        java.lang.ClassNotFoundException: if a serialized class is not
            found by the current classloader.
        java.io.InvalidClassException: if the serialVersionUID for the
            class does not match, usually as a result of a new jar
            version.
        java.io.StreamCorruptedException: if the pickle file has been
            altered or corrupted.

    """

    def __init__(self, file, *args, **kwargs):
        pickle.Unpickler.__init__(self, file, *args, **kwargs)

    def find_class(self, module, cls):
        """Specialization for Java classes.

        We just need to substitute the stub class for a real
        one which points to our decoder instance.
        """
        if cls == "JUnserializer":

            class JUnserializer(object):
                def __call__(self, *args):
                    return _jpype.JClass("org.jpype.pickle.Decoder")().unpack(args[0])

            return JUnserializer
        return pickle.Unpickler.find_class(self, module, cls)
