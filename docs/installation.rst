Local Installation
==================


Python Version
--------------
CapyMOA is tested against Python 3.9, 3.10, and 3.11. Newer versions of python
will likely work, but have not been tested.


Dependencies
------------

Java
~~~~
CapyMOA requires Java to be installed and accessible in your environment. You
can check if Java is installed by running the following command in your terminal:

.. code-block:: sh

    java -version

If Java is not installed, you can download OpenJDK (Open Java Development Kit) 
from `this link <https://openjdk.org/install/>`_, or alternatively the Oracle 
JDK from `this link <https://www.oracle.com/java>`_. Linux users can also
install OpenJDK using their distribution's package manager.

Now that Java is installed, you should see an output similar to the following
when you run the command ``java -version``::

    openjdk version "17.0.9" 2023-10-17
    OpenJDK Runtime Environment (build 17.0.9+8)
    OpenJDK 64-Bit Server VM (build 17.0.9+8, mixed mode)


PyTorch (Optional)
~~~~~~~~~~~~~~~~~~
The CapyMOA algorithms using deep learning require PyTorch. It
is not installed by default because different versions are required
for different hardware. If you want to use these algorithms, follow the
instructions `here <https://pytorch.org/get-started/locally/>`_ to
get the correct version for your hardware.



Install CapyMOA
---------------
Within the activated environment, you can install CapyMOA with pip:

.. code-block:: sh

    pip install capymoa
