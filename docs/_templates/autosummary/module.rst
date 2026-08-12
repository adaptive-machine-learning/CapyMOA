{{ ('``' ~ name  ~ '``') | underline}}

.. currentmodule:: {{ fullname }}

.. automodule:: {{ fullname }}

{% block modules %}
{%- if modules %}
Modules
-------

.. autosummary::
   :toctree:
   :recursive:
{% for item in modules %}
   {{ item }}
{%- endfor %}
{%- endif %}
{% endblock %}

{% block classes %}
{%- set classes = classes | reject("in", attributes) | list %}
{%- if classes %}
Classes
-------

.. autosummary::
   :toctree:
   :nosignatures:
{% for item in classes %}
   {{ item }}
{%- endfor %}
{%- endif %}
{%- endblock %}

{% block attributes %}
{%- if attributes %}
Module Attributes
-----------------

.. autosummary::
{% for item in attributes %}
   {{ item }}
{%- endfor %}

{% for item in attributes %}
.. autodata:: {{ item }}
{% endfor %}
{%- endif %}
{%- endblock %}

{% block functions %}
{%- if functions %}
Functions
---------

.. autosummary::
   :nosignatures:
{% for item in functions %}
   {{ item }}
{%- endfor %}

{% for item in functions %}
.. autofunction:: {{ item }}
{% endfor %}
{%- endif %}
{%- endblock %}

{% block exceptions %}
{%- if exceptions %}
Exceptions
----------

.. autosummary::
{% for item in exceptions %}
   {{ item }}
{%- endfor %}

{% for item in exceptions %}
.. autoexception:: {{ item }}
{% endfor %}
{%- endif %}
{%- endblock %}
