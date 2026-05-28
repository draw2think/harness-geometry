# Arc Command

> Source: `code/docs/geogebra-manual-main-en/modules/ROOT/pages/commands/Arc.adoc`
> Page slug: `commands/Arc`

## Syntax

`Arc( <Circle>, <Point M>, <Point N> )`

Returns the directed arc (counterclockwise) of the given circle, with endpoints M and N.

`Arc( <Ellipse>, <Point M>, <Point N> )`

Returns the directed arc (counterclockwise) of the given ellipse, with endpoints M and N.

`Arc( <Circle>, <Parameter Value>, <Parameter Value> )`

Returns the circle arc of the given circle, whose endpoints are identified by the specified values of the parameter.

> **Note**
>
> Internally the following parametric forms are used: Circle: (*r* cos(*t*), *r* sin(*t*)) where *r* is the circle's radius.

`Arc( <Ellipse>, <Parameter Value>, <Parameter Value> )`

Returns the circle arc of the given ellipse, whose endpoints are identified by the specified values of the parameter.

> **Note**
>
> Internally the following parametric forms are used: Ellipse: (*a* cos(*t*), *b* sin(*t*)) where *a* and *b* are the lengths of the semimajor and semiminor axes.

> **Note**
>
> See also [CircumcircularArc](../../../../code/docs/geogebra-manual-main-en/modules/ROOT/pages/commands/CircumcircularArc.adoc) command.
