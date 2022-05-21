# gossipy documentation

```{admonition} Under construction!
:class: warning

This documentation is currently under construction.
```

Some awesome **text** here!

```{eval-rst}
.. toctree::
   :caption: Documentation
   :maxdepth: 2

   getting_started
   api/index
```


```python
from a import b
c = "string"
```

```{figure} ./imgs/gl_algorithm.png
:name: test label

This is a caption for the image
```

```{eval-rst}
.. mermaid::

   sequenceDiagram
      participant Alice
      participant Bob
      Alice->John: Hello John, how are you?
      loop Healthcheck
          John->John: Fight against hypochondria
      end
      Note right of John: Rational thoughts <br/>prevail...
      John-->Alice: Great!
      John->Bob: How about you?
      Bob-->John: Jolly good!
```

```{seealso}
For an introduction to using MyST with Sphinx, see []().
```