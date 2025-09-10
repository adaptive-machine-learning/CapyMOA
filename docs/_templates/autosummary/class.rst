{{ name | escape | underline}}

.. currentmodule:: {{ module }}
.. autoclass:: {{ objname }}
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__, __call__, __iter__, __next__
   :member-order: groupwise
   {%- if module not in inherited_members_module_denylist %}
   :inherited-members:
   {% endif %}
