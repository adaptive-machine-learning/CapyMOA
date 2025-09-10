{{ name | escape | underline}}

..  currentmodule:: {{ fullname }}

..  automodule:: {{ fullname }}
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
{%- if classes %}
Classes
-------

..  autosummary::
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

{%- for item in attributes %}
.. autodata:: {{ item }}
{%- endfor %}
{%- endif %}
{%- endblock %}
{% block functions %}
{%- if functions %}
Functions
---------

{%- for item in functions %}
.. autofunction:: {{ item }}
{%- endfor %}
{%- endif %}
{%- endblock %}

{% block exceptions %}
{%- if exceptions %}
Exceptions
----------

{%- for item in exceptions %}
.. autoexception:: {{ item }}
{%- endfor %}
{%- endif %}
{%- endblock %}
